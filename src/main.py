import argparse
import pickle
import pprint
import sys
from collections import OrderedDict

import numpy as np

from data.breakfast import BreakfastCorpus
from data.corpus import Datasplit
from data.crosstask import CrosstaskCorpus
from models.framewise import FramewiseGaussianMixture, FramewiseDiscriminative
from models.model import Model, add_training_args
from utils.logger import logger

CLASSIFIERS = {
    'framewise_discriminative': FramewiseDiscriminative,
    'framewise_gaussian_mixture': FramewiseGaussianMixture,
}


def add_data_args(parser):
    parser.add_argument('--dataset', choices=['crosstask', 'breakfast'], default='crosstask')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--features', choices=['raw', 'pca'], default='raw')
    parser.add_argument('--remove_background', action='store_true')
    parser.add_argument('--pca_components_per_group', type=int, default=100)
    parser.add_argument('--crosstask_feature_groups', choices=['i3d', 'resnet', 'audio'], default=['i3d', 'resnet'])


def add_classifier_args(parser):
    parser.add_argument('--classifier', required=True, choices=CLASSIFIERS.keys())
    parser.add_argument('--cuda', action='store_true')
    for name, cls in CLASSIFIERS.items():
        cls.add_args(parser)


def test(model: Model, test_data: Datasplit, test_data_name: str, verbose=True, optimal_assignment=True):
    predictions_by_video = model.predict(test_data)
    prediction_function = lambda video: predictions_by_video[video.name]
    stats = test_data.accuracy_corpus(optimal_assignment,
                                      prediction_function,
                                      prefix=test_data_name,
                                      verbose=verbose)
    return stats


def train(args, train_data: Datasplit, dev_data: Datasplit, split_name, verbose=False, optimal_assignment=True):
    model = CLASSIFIERS[args.classifier].from_args(args, train_data)

    def evaluate_on_data(data, name):
        stats_by_name = test(model, data, name, verbose=verbose, optimal_assignment=optimal_assignment)

        all_mof = np.array([stats['mof'] for stats in stats_by_name.values()])
        sum_mof = all_mof.sum(axis=0)
        right, total = sum_mof
        return float(right) / total

    models_by_epoch = {}
    dev_mof_by_epoch = {}

    def callback_fn(epoch, stats):
        train_mof = evaluate_on_data(train_data, 'train')
        dev_mof = evaluate_on_data(dev_data, 'dev')
        log_str = '{}\tepoch {:2d}'.format(split_name, epoch)
        for stat, value in stats.items():
            if isinstance(value, float):
                log_str += '\t{} {:.4f}'.format(stat, value)
            else:
                log_str += '\t{} {}'.format(stat, value)
        log_str += '\ttrain mof {:.4f}\tdev mof {:.4f}'.format(train_mof, dev_mof)
        logger.debug(log_str)
        models_by_epoch[epoch] = pickle.dumps(model)
        dev_mof_by_epoch[epoch] = dev_mof

    model.fit(train_data, callback_fn)

    if dev_mof_by_epoch:
        best_dev_epoch, best_dev_mof = max(dev_mof_by_epoch.items(), key=lambda t: t[1])
        logger.debug("best dev mov {:.4f} in epoch {}".format(best_dev_mof, best_dev_epoch))
        return pickle.loads(models_by_epoch[best_dev_epoch])
    else:
        return model


def make_data_splits(args):
    # split_name -> (train_data, test_data)
    splits = OrderedDict()

    if args.dataset == 'crosstask':
        pass
        if args.features == 'pca':
            max_components = 200
            assert args.pca_components_per_group <= max_components
            feature_root = 'data/crosstask/crosstask_processed/crosstask_primary_pca-{}_with-bkg_by-task'.format(max_components)
            dimensions_per_feature_group = {
                feature_group: args.pca_components_per_group
                for feature_group in args.crosstask_feature_groups
            }
        else:
            feature_root = 'data/crosstask/crosstask_features'
            dimensions_per_feature_group = None
        for task_id in CrosstaskCorpus.TASK_IDS_BY_SET['primary']:
            corpus = CrosstaskCorpus(
                release_root="data/crosstask/crosstask_release",
                feature_root=feature_root,
                dimensions_per_feature_group=dimensions_per_feature_group,
            )
            corpus._cache_features = True

            splits['{}_val'.format(task_id)] = (
                corpus.get_datasplit(remove_background=args.remove_background,
                                     task_sets=['primary'],
                                     task_ids=[task_id],
                                     split='train'),
                corpus.get_datasplit(remove_background=args.remove_background,
                                     task_sets=['primary'],
                                     task_ids=[task_id],
                                     split='val'),
            )
    elif args.dataset == 'breakfast':
        corpus = BreakfastCorpus('data/breakfast/mapping.txt',
                                 'data/breakfast/reduced_fv_64',
                                 'data/breakfast/BreakfastII_15fps_qvga_sync')
        corpus._cache_features = True

        all_splits = list(sorted(BreakfastCorpus.DATASPLITS.keys()))
        for heldout_split in all_splits:
            splits[heldout_split] = (
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[sp for sp in all_splits if sp != heldout_split]),
                corpus.get_datasplit(remove_background=args.remove_background,
                                     splits=[heldout_split]),
            )
    else:
        raise NotImplementedError("invalid dataset {}".format(args.dataset))

    return splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_data_args(parser)
    add_classifier_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    print(' '.join(sys.argv))

    pprint.pprint(vars(args))

    stats_by_split_and_task = {}

    for split_name, (train_data, test_data) in make_data_splits(args).items():
        print(split_name)
        model = train(args, train_data, test_data, split_name)

        stats_by_task = test(model, test_data, split_name)
        for task, stats in stats_by_task.items():
            stats_by_split_and_task["{}_{}".format(split_name, task)] = stats
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

