import os
import argparse
import pickle
import pprint
import sys
from collections import OrderedDict

import numpy as np

from data.breakfast import BreakfastCorpus
from data.corpus import Datasplit
from data.crosstask import CrosstaskCorpus
from models.framewise import FramewiseGaussianMixture, FramewiseDiscriminative, FramewiseBaseline
from models.sequential import SequentialDiscriminative
from models.model import Model, add_training_args
from models.semimarkov.semimarkov import SemiMarkovModel
from utils.logger import logger

CLASSIFIERS = {
    'framewise_discriminative': FramewiseDiscriminative,
    'framewise_gaussian_mixture': FramewiseGaussianMixture,
    'framewise_baseline': FramewiseBaseline,
    'semimarkov': SemiMarkovModel,
    'sequential_discriminative': SequentialDiscriminative,
}


def add_serialization_args(parser):
    group = parser.add_argument_group('serialization')
    group.add_argument('--model_output_path')


def add_data_args(parser):
    group = parser.add_argument_group('data')
    group.add_argument('--dataset', choices=['crosstask', 'breakfast'], default='crosstask')
    group.add_argument('--features', choices=['raw', 'pca'], default='raw')
    group.add_argument('--feature_downscale', type=float, default=1.0)
    group.add_argument('--batch_size', type=int, default=5)
    group.add_argument('--remove_background', action='store_true')
    group.add_argument('--pca_components_per_group', type=int, default=100)
    group.add_argument('--pca_no_background', action='store_true')
    group.add_argument('--crosstask_feature_groups', choices=['i3d', 'resnet', 'audio', 'narration'], nargs='+', default=['i3d', 'resnet', 'narration'])

    group.add_argument('--crosstask_training_data', choices=['primary', 'related'], nargs='+', default=['primary'])

    group.add_argument('--mix_tasks', action='store_true', help='train on all tasks simultaneously')

    group.add_argument('--compare_to_prediction_folder', help='root folder containing *_pred.npy and *_true.npy prediction files (for comparison)')

    group.add_argument('--frame_subsample', type=int, default=1, help="interval to subsample frames at (e.g. 10 takes every 10th frame)")

    group.add_argument('--task_specific_steps', action='store_true',
                       help="")


def add_classifier_args(parser):
    group = parser.add_argument_group('classifier')
    group.add_argument('--classifier', required=True, choices=CLASSIFIERS.keys())
    group.add_argument('--training', choices=['supervised', 'unsupervised'], default='supervised')
    group.add_argument('--cuda', action='store_true')
    for name, cls in CLASSIFIERS.items():
        cls.add_args(parser)


def test(args, model: Model, test_data: Datasplit, test_data_name: str, verbose=True):
    if args.training == 'supervised':
        optimal_assignment = False
    else:
        assert args.training == 'unsupervised'
        optimal_assignment = True
    predictions_by_video = model.predict(test_data)
    prediction_function = lambda video: predictions_by_video[video.name]
    stats = test_data.accuracy_corpus(optimal_assignment,
                                      prediction_function,
                                      prefix=test_data_name,
                                      verbose=verbose,
                                      compare_to_folder=args.compare_to_prediction_folder)
    return stats


def train(args, train_data: Datasplit, dev_data: Datasplit, split_name, verbose=False, train_sub_data=None):
    model = CLASSIFIERS[args.classifier].from_args(args, train_data)

    if args.training == 'supervised':
        use_labels = True
        early_stopping_on_dev = True
    else:
        assert args.training == 'unsupervised'
        use_labels = False
        early_stopping_on_dev = False

    def evaluate_on_data(data, name):
        stats_by_name = test(args, model, data, name, verbose=verbose)

        all_mof = np.array([stats['mof'] for stats in stats_by_name.values()])
        sum_mof = all_mof.sum(axis=0)
        right, total = sum_mof
        return float(right) / total

    models_by_epoch = {}
    dev_mof_by_epoch = {}
    stats_by_epoch = {}

    def callback_fn(epoch, stats):
        stats_by_epoch[epoch] = stats
        if train_sub_data is not None:
            train_name = 'train_subset'
            train_mof = evaluate_on_data(train_sub_data, train_name)
        else:
            train_name = 'train'
            train_mof = evaluate_on_data(train_data, train_name)
        dev_mof = evaluate_on_data(dev_data, 'dev')
        log_str = '{}\tepoch {:2d}'.format(split_name, epoch)
        for stat, value in stats.items():
            if isinstance(value, float):
                log_str += '\t{} {:.4f}'.format(stat, value)
            else:
                log_str += '\t{} {}'.format(stat, value)
        log_str += '\t{} mof {:.4f}\tdev mof {:.4f}'.format(train_name, train_mof, dev_mof)
        logger.debug(log_str)
        models_by_epoch[epoch] = pickle.dumps(model)
        dev_mof_by_epoch[epoch] = dev_mof

    model.fit(train_data, use_labels=use_labels, callback_fn=callback_fn)

    if early_stopping_on_dev and dev_mof_by_epoch:
        best_dev_epoch, best_dev_mof = max(dev_mof_by_epoch.items(), key=lambda t: t[1])
        logger.debug("best dev mov {:.4f} in epoch {}".format(best_dev_mof, best_dev_epoch))
        best_model = pickle.loads(models_by_epoch[best_dev_epoch])
    elif stats_by_epoch and 'train_loss' in next(iter(stats_by_epoch.values())):
        best_epoch, best_train_stats = min(stats_by_epoch.items(), key=lambda t: t[1]['train_loss'])
        logger.debug("best train loss {:.4f} in epoch {}".format(best_train_stats['train_loss'], best_epoch))
        best_model = pickle.loads(models_by_epoch[best_epoch])
    else:
        best_model = model

    if args.model_output_path:
        os.makedirs(args.model_output_path, exist_ok=True)
        model_fname = os.path.join(args.model_output_path, '{}.pkl'.format(split_name))
        print("writing model to {}".format(model_fname))
        with open(model_fname, 'wb') as f:
            pickle.dump(best_model, f)

    return best_model


def make_data_splits(args):
    # split_name -> (train_data, test_data)
    splits = OrderedDict()

    if args.dataset == 'crosstask':
        features_contain_background = True
        if args.features == 'pca':
            max_components = 200
            assert args.pca_components_per_group <= max_components
            features_contain_background = not args.pca_no_background
            feature_root = 'data/crosstask/crosstask_processed/crosstask_primary_pca-{}_{}-bkg_by-task'.format(
                max_components,
                "no" if args.pca_no_background else "with",
            )
            dimensions_per_feature_group = {
                feature_group: args.pca_components_per_group
                for feature_group in args.crosstask_feature_groups
            }
        else:
            feature_root = 'data/crosstask/crosstask_features'
            dimensions_per_feature_group = None

        corpus = CrosstaskCorpus(
            release_root="data/crosstask/crosstask_release",
            feature_root=feature_root,
            dimensions_per_feature_group=dimensions_per_feature_group,
            features_contain_background=features_contain_background,
            task_specific_steps=args.task_specific_steps,
            use_secondary='related' in args.crosstask_training_data,
        )
        corpus._cache_features = True
        train_task_sets = args.crosstask_training_data
        test_task_sets = ['primary']
        task_ids = sorted([task_id for task_set in sorted(set(train_task_sets) | set(test_task_sets))
                           for task_id in CrosstaskCorpus.TASK_IDS_BY_SET[task_set]])
        split_names_and_full = [('train', True, train_task_sets), ('train', False, test_task_sets), ('val', True, test_task_sets)]
        if args.mix_tasks:
            splits['all'] = tuple(
                corpus.get_datasplit(remove_background=args.remove_background,
                                     task_sets=task_sets,
                                     task_ids=task_ids,
                                     split=split,
                                     full=full,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale)
                for split, full, task_sets in split_names_and_full
            )
        else:
            for task_id in task_ids:
                splits['{}_val'.format(task_id)] = tuple(
                    corpus.get_datasplit(remove_background=args.remove_background,
                                         task_sets=task_sets,
                                         task_ids=[task_id],
                                         split=split,
                                         full=full,
                                         subsample=args.frame_subsample,
                                         feature_downscale=args.feature_downscale)
                    for split, full, task_sets in split_names_and_full
                )
    elif args.dataset == 'breakfast':
        if args.features == 'pca':
            max_components = 64
            assert args.pca_components_per_group == max_components
            features_contain_background = not args.pca_no_background
            assert features_contain_background # not implemented!
            feature_root = 'data/breakfast/breakfast_processed/breakfast_pca-{}_{}-bkg_by-task'.format(
                max_components,
                "no" if args.pca_no_background else "with",
            )
        else:
            feature_root = 'data/breakfast/reduced_fv_64'
        corpus = BreakfastCorpus(mapping_file='data/breakfast/mapping.txt',
                                 feature_root=feature_root,
                                 label_root='data/breakfast/BreakfastII_15fps_qvga_sync',
                                 task_specific_steps=args.task_specific_steps)
        corpus._cache_features = True

        all_splits = list(sorted(BreakfastCorpus.DATASPLITS.keys()))
        for heldout_split in all_splits:
            splits[heldout_split] = (
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[sp for sp in all_splits if sp != heldout_split],
                                     full=True,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale),
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[sp for sp in all_splits if sp != heldout_split],
                                     full=True,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale), # has issue with some tasks being dropped if we pass full=False
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[heldout_split],
                                     full=True,
                                     subsample=args.frame_subsample,
                                     feature_downscale=args.feature_downscale),
            )
    else:
        raise NotImplementedError("invalid dataset {}".format(args.dataset))

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    add_serialization_args(parser)
    add_data_args(parser)
    add_classifier_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    print(' '.join(sys.argv))

    pprint.pprint(vars(args))

    stats_by_split_and_task = {}

    stats_by_split_by_task = {}

    for split_name, (train_data, train_sub_data, test_data) in make_data_splits(args).items():
        print(split_name)
        model = train(args, train_data, test_data, split_name, train_sub_data=train_sub_data)

        stats_by_task = test(args, model, test_data, split_name)
        stats_by_split_by_task[split_name] = {}
        for task, stats in stats_by_task.items():
            stats_by_split_and_task["{}_{}".format(split_name, task)] = stats
            stats_by_split_by_task[split_name][task] = stats
        print()


    def divide(d):
        divided = {}
        for key, vals in d.items():
            assert len(vals) == 2
            divided[key] = float(vals[0]) / vals[1]
        return divided


    print()
    pprint.pprint(stats_by_split_and_task)

    print()
    pprint.pprint({k: divide(d) for k, d in stats_by_split_and_task.items()})

    summed_across_tasks = {}
    divided_averaged_across_tasks = {}

    sum_within_split_averaged_across_splits = {}

    for key in next(iter(stats_by_split_and_task.values())):
        arrs = np.array([d[key] for d in stats_by_split_and_task.values()])
        summed_across_tasks[key] = np.sum(arrs, axis=0)

        divided_averaged_across_tasks[key] = np.mean([
            divide(d)[key] for d in stats_by_split_and_task.values()
        ])

    print()

    summed_across_tasks_divided = divide(summed_across_tasks)

    print("summed across tasks:")
    pprint.pprint(summed_across_tasks_divided)
    print()
    print("averaged across tasks:")
    pprint.pprint(divided_averaged_across_tasks)
    print()
    print("averaged across splits:")
    pprint.pprint(sum_within_split_averaged_across_splits)

