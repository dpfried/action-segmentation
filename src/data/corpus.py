# modified from slim_mallow by Anna Kukleva, https://github.com/Annusha/slim_mallow

import os
import copy

import random
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from evaluation.accuracy import Accuracy
from evaluation.f1 import F1Score
from utils.logger import logger
from utils.utils import nested_dict_map

FEATURE_LABEL_MISMATCH_TOLERANCE = 50

WARN_ON_MISMATCH = False


class Video(object):
    def __init__(self, feature_root, K, remove_background, *, nonbackground_timesteps=None,
                 gt=None, gt_with_background=None, name='', cache_features=False, has_label=True,
                 features_contain_background=True):
        """
        Args:
            feature_root (str): path to video representation
            K (int): number of subactivities in current task
            reset (bool): necessity of holding features in each instance
            gt (arr): ground truth labels
            gt_with_background (arr): ground truth labels with background (0) label
            name (str): short name without any extension
        """
        self.iter = 0
        self._feature_root = feature_root
        self._K = K
        self.name = name
        self._cache_features = cache_features
        self._has_label = has_label
        self._features_contain_background = features_contain_background

        assert name

        if remove_background:
            assert has_label
            assert nonbackground_timesteps is not None
            assert len(nonbackground_timesteps) == len(gt)
        self._remove_background = remove_background
        self._nonbackground_timesteps = nonbackground_timesteps

        # self._likelihood_grid = None
        # self._valid_likelihood = None
        # self._theta_0 = 0.1
        # self._subact_i_mask = np.eye(self._K)
        self._features = None
        # self.global_start = start
        # self.global_range = None

        self._n_frames = None

        self._gt = gt if gt is not None else []
        self._gt_unique = np.unique(self._gt)
        self._gt_with_background = gt_with_background

        self._updated_length = False

        # if features is None:
        #     features = self.load_features(self.path)
        # self._set_features(features)

        # self._warned_length = False

        # ordering, init with canonical ordering
        # self._pi = list(range(self._K))
        # self.inv_count_v = np.zeros(self._K - 1)
        # subactivity per frame
        # self._z = []
        # self._z_idx = []
        # self._init_z_framewise()

        # self.fg_mask = np.ones(self.n_frames, dtype=bool)
        # if self._with_bg:
        #     self._init_fg_mask()

        # self._subact_count_update()

        self.segmentation = {'gt': (self._gt, None)}

    def load_features(self):
        raise NotImplementedError("should be implemented by subclasses")

    def features(self):
        self._check_truncation()
        if self._cache_features:
            if self._features is None:
                self._features = self._process_features(self.load_features())
            features = self._features
        else:
            features = self._process_features(self.load_features())
        return features

    def n_frames(self):
        return self._n_frames

    def _check_truncation(self):
        if not self._has_label:
            return
        n_frames = self.n_frames()
        if n_frames is None:
            # TODO: ugh
            self._process_features(self.load_features())
            n_frames = self.n_frames()
        assert n_frames is not None
        if not self._updated_length and (len(self._gt_with_background) != n_frames or not self._features_contain_background):
            self._updated_length = True
            if WARN_ON_MISMATCH:
                print(self.name, '# of gt and # of frames does not match %d / %d' %
                      (len(self._gt_with_background), n_frames))

            assert len(self._gt_with_background) - n_frames <= FEATURE_LABEL_MISMATCH_TOLERANCE, "len(self._gt_with_background) = {}, n_frames = {}".format(len(self._gt_with_background), n_frames)
            min_n = min(len(self._gt_with_background), n_frames)
            # self._gt = self._gt[:min_n]
            # self._gt_with_background = self._gt_with_background[:min_n]
            self._n_frames = min_n
            # invalidate cache
            self._features = None

    def gt(self):
        self._check_truncation()
        if self._remove_background:
            tnb = self._truncated_nonbackground_timesteps()
            gt = self._gt_with_background[:self.n_frames()]
            new_gt = []
            for ix in tnb:
                new_gt.append(gt[ix])
            gt = new_gt
            assert len(gt) == len(tnb)
        else:
            gt = self._gt[:self.n_frames()]
        return gt

    def gt_with_background(self):
        self._check_truncation()
        return self._gt_with_background[:self.n_frames()]

    def _truncated_nonbackground_timesteps(self):
        return [t for t in self._nonbackground_timesteps if t < self.n_frames()]

    def _process_features(self, features):
        if self._n_frames is None:
            if self._features_contain_background:
                self._n_frames = features.shape[0]
            else:
                self._n_frames = len(self._gt_with_background)
        if not self._features_contain_background:
            return features
        # zeros = 0
        # for i in range(10):
        #     if np.sum(features[:1]) == 0:
        #         features = features[1:]
        #         zeros += 1
        #     else:
        #         break
        # self._gt = self._gt[zeros:]
        # self._gt_with_background = self._gt_with_background[zeros:]
        features = features[:self.n_frames()]
        if self._remove_background:
            features = features[self._truncated_nonbackground_timesteps()]
        return features

    # def _init_z_framewise(self):
    #     """Init subactivities uniformly among video frames"""
    #     # number of frames per activity
    #     step = math.ceil(self.n_frames / self._K)
    #     modulo = self.n_frames % self._K
    #     for action in range(self._K):
    #         # uniformly distribute remainder per actions if n_frames % K != 0
    #         self._z += [action] * (step - 1 * (modulo <= action) * (modulo != 0))
    #     self._z = np.asarray(self._z, dtype=int)
    #     try:
    #         assert len(self._z) == self.n_frames
    #     except AssertionError:
    #         logger.error('Wrong initialization for video %s', self.path)

    # FG_MASK
    # def _init_fg_mask(self):
    #     indexes = [i for i in range(self.n_frames) if i % 2]
    #     self.fg_mask[indexes] = False
    #     # todo: have to check if it works correctly
    #     # since it just after initialization
    #     self._z[self.fg_mask == False] = -1

    # def update_indexes(self, total):
    #     self.global_range = np.zeros(total, dtype=bool)
    #     self.global_range[self.global_start: self.global_start + self.n_frames] = True

    # def reset(self):
    #     """If features from here won't be in use anymore"""
    #     self._features = None

    # def z(self, pi=None):
    #     """Construct z (framewise label assignments) from ordering and counting.
    #     Args:
    #         pi: order, if not given the current one is used
    #     Returns:
    #         constructed z out of indexes instead of actual subactivity labels
    #     """
    #     self._z = []
    #     self._z_idx = []
    #     if pi is None:
    #         pi = self._pi
    #     for idx, activity in enumerate(pi):
    #         self._z += [int(activity)] * self.a[int(activity)]
    #         self._z_idx += [idx] * self.a[int(activity)]
    #     if opt.bg:
    #         z = np.ones(self.n_frames, dtype=int) * -1
    #         z[self.fg_mask] = self._z
    #         self._z = z[:]
    #         z[self.fg_mask] = self._z_idx
    #         self._z_idx = z[:]
    #     assert len(self._z) == self.n_frames
    #     return np.asarray(self._z_idx)


class Datasplit(Dataset):
    def __init__(self, corpus, remove_background, full=True):
        self._corpus = corpus
        self._remove_background = remove_background
        self._full = full

        # logger.debug('%s  subactions: %d' % (subaction, self._K))
        self.return_stat = {}

        self._videos_by_task = {}
        # init with ones for consistency with first measurement of MoF
        # self._subact_counter = np.ones(self._K)
        # number of gaussian in each mixture
        self._gt2label = None
        self._label2gt = {}

        # FG_MASK
        # self._total_fg_mask = None

        # multiprocessing for sampling activities for each video
        self._features_by_task = None
        self._embedded_feat = None

        self.groundtruth = None
        self._K_by_task = None
        self._load_ground_truth_and_videos(remove_background)
        assert self.groundtruth is not None, "_load_ground_truth_and_videos didn't set groundtruth"
        assert len(self._videos_by_task) != 0, "_load_ground_truth_and_videos didn't load any task's videos"
        assert self._K_by_task is not None, "_load_ground_truth_and_videos didn't set _K_by_task"

        self._tasks_and_video_names = list(sorted([
            (task_name, video_name)
            for task_name, vid_dict in self._videos_by_task.items()
            for video_name in vid_dict
        ]))

        # logger.debug('min: %f  max: %f  avg: %f' %
        #              (np.min(self._features),
        #               np.max(self._features),
        #               np.mean(self._features)))

    def batch_sampler(self, batch_size=1, batch_by_task=True, shuffle=False):
        return BatchSampler(self, batch_size=batch_size, batch_by_task=batch_by_task, shuffle=shuffle)

    @property
    def corpus(self):
        return self._corpus

    @property
    def remove_background(self):
        return self._remove_background

    def __len__(self):
        return len(self._tasks_and_video_names)

    def __getitem__(self, task_and_video_name, wrap_torch=True):
        task_name, video_name = task_and_video_name
        video_obj: Video = self._videos_by_task[task_name][video_name]

        # num_timesteps = torch_features.size(0)
        features = video_obj.features()
        task_indices = self.groundtruth.indices_by_task[task_name]
        if self.remove_background:
            task_indices = set(task_indices) - {self.groundtruth._corpus._background_index}
        task_indices = sorted(task_indices)
        gt_single = [gt_t[0] for gt_t in video_obj.gt()]

        if wrap_torch:
            features = torch.from_numpy(features).float()
            task_indices = torch.LongTensor(task_indices)
            gt_single = torch.LongTensor(gt_single)
        else:
            task_indices = list(task_indices)
        data = {
            'task_name': task_name,
            'video_name': video_name,
            'features': features,
            'gt': video_obj.gt(),
            'gt_single': gt_single,
            'gt_with_background': video_obj.gt_with_background(),
            'task_indices': task_indices,
        }
        return data

    def _get_by_index(self, index, wrap_torch=True):
        task_name, video_name = self._tasks_and_video_names[index]
        return self.__getitem__((task_name, video_name), wrap_torch)

    @property
    def feature_dim(self):
        return self._get_by_index(0)['features'].size(1)

    def _load_ground_truth_and_videos(self, remove_background):
        raise NotImplementedError("subclasses should implement _load_ground_truth")

    def accuracy_corpus(self, optimal_assignment: bool, prediction_function, prefix='', verbose=True, compare_to_folder=None):
        """Calculate metrics as well with previous correspondences between
        gt labels and output labels"""
        stats_by_task = {}
        for task in self._videos_by_task:
            if verbose:
                logger.debug("computing accuracy for task {}".format(task))
            accuracy = Accuracy(verbose=verbose, corpus=self._corpus)

            f1_score = F1Score(K=self._K_by_task[task], n_videos=len(self._videos_by_task[task]), verbose=verbose)
            long_gt = []
            long_pr = []

            if compare_to_folder is not None:
                compare_accuracy = Accuracy(verbose=verbose, corpus=self._corpus)
                compare_long_gt = []
                compare_long_pr = []

            # long_gt_onhe0 = []
            self.return_stat = {}

            if compare_to_folder is not None:
                task_mapping = {}

            def load_predictions(video_name):
                if os.path.exists(os.path.join(compare_to_folder, "{}_y_true.npy".format(video_name))):
                    y_true = np.load(os.path.join(compare_to_folder, "{}_y_true.npy".format(video_name)))
                    y_pred = np.load(os.path.join(compare_to_folder, "{}_y_pred.npy".format(video_name)))
                    return {
                        'y_true': y_true,
                        'y_pred': y_pred,
                    }
                else:
                    import json
                    with open(os.path.join(compare_to_folder, "{}.json".format(video_name))) as f:
                        pred_data = json.load(f)
                        return {
                            key: np.array(val)
                            for key, val in pred_data.items()
                        }


            for video_name, video in self._videos_by_task[task].items():
                # long_gt += list(video._gt_with_0)
                # long_gt_onhe0 += list(video._gt)
                long_gt += list(video.gt())
                long_pr += list(prediction_function(video))

                if compare_to_folder is not None:
                    # break ties with background, or earlier steps
                    pred_data = load_predictions(video_name)
                    y_true = pred_data['y_true']
                    y_pred = pred_data['y_pred']
                    trues = y_true.argmax(axis=1)
                    preds = y_pred.argmax(axis=1)

                    assert len(trues) == len(video.gt())
                    for t, g in zip(trues, video.gt()):
                        g = g[0]
                        if t in task_mapping:
                            assert task_mapping[t] == g
                        else:
                            task_mapping[t] = g

            if compare_to_folder is not None:
                for video_name, video in self._videos_by_task[task].items():
                    pred_data = load_predictions(video_name)
                    y_true = pred_data['y_true']
                    y_pred = pred_data['y_pred']
                    # break ties with background, or earlier steps
                    trues = y_true.argmax(axis=1)
                    preds = y_pred.argmax(axis=1)
                    # y_true_rc = pred_data['y_true_rc']
                    # y_pred_rc = pred_data['y_pred_rc']
                    # trues = y_true_rc.argmax(axis=1)
                    # preds = y_pred_rc.argmax(axis=1)

                    compare_long_gt += [[task_mapping[t]] for t in trues]
                    compare_long_pr += [task_mapping[p] for p in preds]

            accuracy.gt_labels = long_gt
            accuracy.predicted_labels = long_pr

            named_accuracies = [('this_run', accuracy)]

            if compare_to_folder is not None:
                compare_accuracy.gt_labels = compare_long_gt
                compare_accuracy.predicted_labels = compare_long_pr
                named_accuracies.append(('comparison', compare_accuracy))

            for acc_name, acc in named_accuracies:
                # if opt.bg:
                #     # enforce bg class to be bg class
                #     acc.exclude[-1] = [-1]
                # if not opt.zeros and opt.dataset == 'bf': #''Breakfast' in opt.dataset_root:
                #     # enforce to SIL class assign nothing
                #     acc.exclude[0] = [-1]

                old_mof, total_fr = acc.mof(optimal_assignment, old_gt2label=self._gt2label)
                if acc_name == 'this_run':
                    self._gt2label = acc._gt2cluster
                    self._label2gt = {}
                    for key, val in self._gt2label.items():
                        try:
                            self._label2gt[val[0]] = key
                        except IndexError:
                            pass
                acc_cur = acc.mof_val()
                if verbose:
                    logger.debug('%s Task: %s' % (prefix, task))
                    logger.debug('%s MoF val: ' % prefix + str(acc_cur))
                    logger.debug('%s previous dic -> MoF val: ' % prefix + str(float(old_mof) / total_fr))

                acc.mof_classes()
                acc.iou_classes()

            self.return_stat = accuracy.stat()

            f1_score.set_gt(long_gt)
            f1_score.set_pr(long_pr)
            f1_score.set_gt2pr(self._gt2label)
            # if opt.bg:
            #     f1_score.set_exclude(-1)
            f1_score.f1()

            for key, val in f1_score.stat().items():
                self.return_stat[key] = val

            for video_name, video in self._videos_by_task[task].items():
                video.segmentation[video.iter] = (prediction_function(video), self._label2gt)

            stats = accuracy.stat()
            if compare_to_folder is not None:
                comparison_stats = compare_accuracy.stat()
                stats['comparison_mof'] = comparison_stats['mof']
                stats['comparison_mof_bg'] = comparison_stats['mof_bg']
                stats['comparison_mof_non_bg'] = comparison_stats['mof_non_bg']

            stats_by_task[task] = accuracy.stat()
        return stats_by_task

    # def resume_segmentation(self):
    #     for video in self._videos:
    #         video.iter = self.iter
    #         video.resume()
    #     self._count_subact()

class BatchSampler(Sampler):
    def __init__(self, datasplit: Datasplit, batch_size: int, batch_by_task: bool, shuffle: bool, seed=1):
        self.datasplit = datasplit
        self.batch_size = batch_size
        self.batch_by_task = batch_by_task
        self.shuffle = shuffle
        self.seed = seed

        self.batches = []
        task_names = list(sorted(self.datasplit._videos_by_task.keys()))

        videos_by_task = {
            task: list(sorted(videos))
            for task, videos in self.datasplit._videos_by_task.items()
        }
        if self.shuffle:
            state = random.Random(self.seed)
            state.shuffle(task_names)
            for videos in videos_by_task.items():
                state.shuffle(videos)

        for task in task_names:
            videos = videos_by_task[task]
            for i in range(0, len(videos), self.batch_size):
                self.batches.append([(task, video) for video in videos[i:i+self.batch_size]])

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class Corpus(object):

    def __init__(self, background_labels, cache_features=False):
        """
        Args:
            K: number of possible subactions in current dataset (TODO: this disappeared)
            subaction: current name of complex activity
        """
        self.label2index = {}
        self.index2label = {}

        self._cache_features= cache_features

        self._background_labels = background_labels
        self._background_indices = list(range(len(background_labels)))
        for label, index in zip(background_labels, self._background_indices):
            self.label2index[label] = index
            self.index2label[index] = label
        self._labels_frozen = False
        self._load_mapping()
        self._labels_frozen = True


    @property
    def n_classes(self):
        return len(self.label2index)

    def _index(self, label):
        if label not in self.label2index:
            assert not self._labels_frozen, "trying to index {} after index has been frozen".format(label)
            label_idx = len(self.label2index)
            self.label2index[label] = label_idx
            self.index2label[label_idx] = label
        else:
            label_idx = self.label2index[label]
        return label_idx

    def _load_mapping(self):
        raise NotImplementedError("subclasses should implement _load_mapping")

    def get_datasplit(self, remove_background, full=True) -> Datasplit:
        raise NotImplementedError("subclasses should implement get_datasplit")

    # def _count_subact(self):
    #     self._subact_counter = np.zeros(self._K)
    #     for video in self._videos:
    #         self._subact_counter += video.a


class GroundTruth(object):
    def __init__(self, corpus: Corpus, task_names, remove_background):
        self._corpus = corpus
        self._task_names = task_names
        self._remove_background = remove_background

        self.gt_by_task = {}
        self.gt_with_background_by_task = {}
        self.order_by_task = {}
        self.order_with_background_by_task = {}

        self.indices_by_task = None

        self.nonbackground_timesteps_by_task = {}
        self.load_gt_and_remove_background()

    def _load_gt(self):
        raise NotImplementedError("_load_gt")

    def load_gt_and_remove_background(self):
        self._load_gt()
        self.gt_with_background_by_task = self.gt_by_task
        # print(list(gt_with_0.keys()))
        self.order_with_background_by_task = self.order_by_task

        if self._remove_background:
            self.remove_background()

        self.indices_by_task = {}
        for task, gt_dict in self.gt_by_task.items():
            label_set = set()
            for vid, gt in gt_dict.items():
                for gt_t in gt:
                    label_set.update(gt_t)
            self.indices_by_task[task] = list(sorted(label_set))

    def remove_background(self):
        self.gt_with_background_by_task = copy.deepcopy(self.gt_by_task)
        self.order_with_background_by_task = copy.deepcopy(self.order_by_task)

        def nonbkg_indices(task, video, gt):
            return [t for t, gt_t in enumerate(gt) if gt_t[0] != self._corpus._background_index]

        self.nonbackground_timesteps_by_task = nested_dict_map(self.gt_by_task, nonbkg_indices)

        def rm_bkg_from_indices(task, video, gt):
            nonbackground_indices = self.nonbackground_timesteps_by_task[task][video]
            nbi_set = set(nonbackground_indices)
            new_gt = []
            for ix, val in enumerate(gt):
                if ix in nbi_set:
                    new_gt.append(val)
            gt = new_gt
            # if value[0] == 0:
            #     for idx, val in enumerate(value):
            #         if val:
            #             value[:idx] = val
            #             break
            # if value[-1] == 0:
            #     for idx, val in enumerate(np.flip(value, 0)):
            #         if val:
            #             value[-idx:] = val
            #             break
            assert self._corpus._background_index not in gt
            return gt

        def rm_bkg_from_order(task, video, order):
            return [t for t in order if t[0] != self._corpus._background_index]

        self.gt_by_task = nested_dict_map(self.gt_by_task, rm_bkg_from_indices)
        self.order_by_task = nested_dict_map(self.order_by_task, rm_bkg_from_order)

    # def sparse_gt(self):
    #     for key, val in self.gt.items():
    #         sparse_segm = [i for i in val[::10]]
    #         self.gt[key] = sparse_segm
    #     self.gt_with_background = copy.deepcopy(self.gt)


