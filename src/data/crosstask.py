import re
import os
import math
import numpy as np

import glob

from collections import namedtuple, defaultdict
from data.corpus import Corpus, GroundTruth, Video

from utils.logger import logger
from data.features import grouped_pca

import pickle

CrosstaskTask = namedtuple("CrosstaskTask", ["index", "title", "url", "n_steps", "steps"])

def read_task_info(path):
    tasks = []
    with open(path,'r') as f:
        index = f.readline()
        while index is not '':
            index = int(index.strip())
            title = f.readline().strip()
            url = f.readline().strip()
            n_steps = int(f.readline().strip())
            steps = f.readline().strip().split(',')
            next(f)
            assert n_steps == len(steps)
            tasks.append(CrosstaskTask(index, title, url, n_steps, steps))
            index = f.readline()
    return tasks


def get_vids(path):
    task_vids = {}
    with open(path,'r') as f:
        for line in f:
            task, vid, url = line.strip().split(',')
            task = int(task)
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids


def read_assignment(T, num_steps, path, include_background=False):
    if include_background:
        cols = num_steps + 1
    else:
        cols = num_steps
    Y = np.zeros([T, cols], dtype=np.uint8)
    with open(path,'r') as f:
        for line in f:
            step,start,end = line.strip().split(',')
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            if include_background:
                step = int(step) - 1
            Y[start:end,step] = 1
    if include_background:
        # turn on the background class (col 0) for any row that has no entries
        Y[Y.sum(axis=1) == 0,0] = 1
    return Y

def read_assignment_list(T, num_steps, path):
    # T x (K + 1)
    Y = read_assignment(T, num_steps, path, include_background=True)
    indices = [list(row.nonzero()[0]) for row in Y]
    assert len(indices) == T
    assert max(max(indices_t) for indices_t in indices) <= num_steps
    return indices

def random_split(task_vids, test_tasks, n_train):
    train_vids = {}
    test_vids = {}
    for task,vids in task_vids.items():
        if task in test_tasks and len(vids) > n_train:
            train_vids[task] = np.random.choice(vids,n_train,replace=False).tolist()
            test_vids[task] = [vid for vid in vids if vid not in train_vids[task]]
        else:
            train_vids[task] = vids
    return train_vids, test_vids

class CrosstaskVideo(Video):

    def __init__(self, *args, dimensions_per_feature_group=None, **kwargs):
        self._dimensions_per_feature_group = dimensions_per_feature_group
        super(CrosstaskVideo, self).__init__(*args, **kwargs)

    @classmethod
    def load_grouped_features(cls, feature_root, dimensions_per_feature_group, video_name):
        if dimensions_per_feature_group is None:
            return np.load(os.path.join(feature_root, "{}.npy".format(video_name)))
        else:
            all_feats = []
            for feature_group, dimensions in sorted(dimensions_per_feature_group.items()):
                feat_path = os.path.join(feature_root, feature_group, "{}.npy".format(video_name))
                feats = np.load(feat_path)
                feats = feats[:,:dimensions]
                all_feats.append(feats)
            return np.hstack(all_feats)

    def load_features(self):
        return CrosstaskVideo.load_grouped_features(self._feature_root, self._dimensions_per_feature_group, self.name)

class CrosstaskGroundTruth(GroundTruth):
    BACKGROUND_LABEL = "BKG"

    def __init__(self, k_by_task, t_by_video, release_folder, task_names, remove_background):
        self._k_by_task = k_by_task
        self._t_by_video = t_by_video
        self._release_folder = release_folder
        self._task_names = task_names
        super(CrosstaskGroundTruth, self).__init__(task_names, remove_background, CrosstaskGroundTruth.BACKGROUND_LABEL)

    def _load_gt(self):
        glob_path = os.path.join(self._release_folder, "annotations", "*.csv")
        filenames = glob.glob(glob_path)
        assert filenames, "no filenames found for glob path {}".format(glob_path)
        logger.debug("{} annotation files found".format(len(filenames)))
        for filename in filenames:
            file = os.path.split(filename)[1]
            file_no_ext = os.path.splitext(file)[0]
            splits = file_no_ext.split('_')
            task = int(splits[0])
            if task not in self._task_names:
                continue
            video = '_'.join(splits[1:])
            T = self._t_by_video[video]
            num_steps = self._k_by_task[task]
            gt = read_assignment_list(T, num_steps, filename)
            if task not in self.gt_by_task:
                self.gt_by_task[task] = {}
            self.gt_by_task[task][video] = gt

DATA_SPLITS = ['train', 'val', 'all']

def load_videos_by_task(release_root, split='train'):
    assert split in DATA_SPLITS

    all_videos_by_task = get_vids(os.path.join(release_root, "videos.csv"))
    if split == 'all':
        return all_videos_by_task
    val_videos_by_task = get_vids(os.path.join(release_root, "videos_val.csv"))
    if split == 'val':
        return val_videos_by_task
    val_videos = set(v for vids in val_videos_by_task.keys() for v in vids)
    train_videos_by_task = {
        task: v for task, vids in all_videos_by_task
        for v in vids
        if v not in val_videos
    }

    assert split == 'train'
    return train_videos_by_task


def corpora_by_task(release_root, feature_root, remove_background, task_sets=None, full=True, split='train', task_ids=None):
    if task_sets is None:
        task_sets = list(CrosstaskCorpus.TASK_SET_PATHS.keys())
    if task_ids is None:
        task_ids = [
            task_id for task_set in task_sets
            for task_id in CrosstaskCorpus.TASK_IDS_BY_SET[task_set]
        ]
    return {
        task_id: CrosstaskCorpus(release_root, feature_root, remove_background,
                                 task_sets=task_sets, full=full, split=split, task_ids=[task_id])
        for task_id in task_ids
    }


class CrosstaskCorpus(Corpus):
    TASK_SET_PATHS = {
        'primary': 'tasks_primary.txt',
        'related': 'tasks_related.txt',
    }

    TASK_IDS_BY_SET = {
        'primary': [16815, 23521, 40567, 44047, 44789, 53193, 59684, 71781, 76400, 77721, 87706, 91515, 94276, 95603, 105222, 105253, 109972, 113766],
        'related': [1373, 11138, 14133, 16136, 16323, 20880, 20898, 23524, 26618, 29477, 30744, 31438, 34938, 34967, 40566, 40570, 40596, 40610, 41718, 41773, 41950, 42901, 44043, 50348, 51659, 53195, 53204, 57396, 67160, 68268, 72954, 75501, 76407, 76412, 77194, 81790, 83956, 85159, 89899, 91518, 91537, 91586, 93376, 93400, 96127, 96366, 97633, 100901, 101028, 103832, 105209, 105259, 105762, 106568, 106686, 108098, 109761, 110266, 113764, 114508, 118421, 118779, 118780, 118819, 118831],
    }
    
    def __init__(self, release_root, feature_root, remove_background, task_sets=None, full=True, split='train', task_ids=None,
                 dimensions_per_feature_group=None):
        self._release_root = release_root
        self._feature_root = feature_root
        self._dimensions_per_feature_group = dimensions_per_feature_group
        if task_sets is None:
            task_sets = list(CrosstaskCorpus.TASK_SET_PATHS.keys())

        assert all(ts in CrosstaskCorpus.TASK_SET_PATHS.keys() for ts in task_sets)
        tasks = [
            task for ts in task_sets
            for task in read_task_info(os.path.join(release_root, CrosstaskCorpus.TASK_SET_PATHS[ts]))
            if task_ids is None or task.index in task_ids
        ]
        task_indices = list(sorted(set([task.index for task in tasks])))
        self._task_names = task_indices

        video_names_by_task = {
            task: videos
            for task, videos in load_videos_by_task(release_root, split=split).items()
            if task in task_indices
        }

        assert len(video_names_by_task) != 0, "no tasks found with task_sets {}, split {}, and release_directory {}".format(
            task_sets, split, release_root
        )

        video_names = list(sorted(set(video for videos in video_names_by_task.values() for video in videos)))
        assert len(video_names) != 0, "no videos found with task_sets {}, split {}, and release_directory {}".format(
            task_sets, split, release_root
        )

        logger.debug("{} tasks found with task_sets {}, split {}".format(len(video_names_by_task), task_sets, split))
        logger.debug("{} videos found with task_sets {}, split {}".format(len(video_names), task_sets, split))

        self._tasks = tasks
        self._video_names_by_task = video_names_by_task

        self._save_frame_counts = (split == 'all' and set(self.TASK_SET_PATHS.keys()) == set(task_sets))

        super(CrosstaskCorpus, self).__init__(task_indices, remove_background=remove_background, full=full)

    def _load_ground_truth_and_videos(self, remove_background):
        self._K_by_task = {
            task.index: len(task.steps)
            for task in self._tasks
        }
        # features_by_task_and_video = {}

        t_by_video_path = os.path.join(self._release_root, "frame_counts.pkl")

        if os.path.exists(t_by_video_path):
            with open(t_by_video_path, 'rb') as f:
                t_by_video = pickle.load(f)
        else:
            logger.debug("creating frame counts")
            t_by_video = {}

            for task_name in self._task_names:
                logger.debug(task_name)
                for video in self._video_names_by_task[task_name]:
                    feats = CrosstaskVideo.load_grouped_features(
                        self._feature_root, self._dimensions_per_feature_group, video
                    )

                    # features_by_task_and_video[(task_name, video)] = feats

                    T = feats.shape[0]
                    if video in t_by_video:
                        assert t_by_video[video] == T, "mismatch in timesteps from features for video {}. stored: {}; new {}".format(video, t_by_video[video], T)
                    t_by_video[video] = T
            if self._save_frame_counts:
                logger.debug("saving to {}".format(t_by_video_path))
                with open(t_by_video_path, 'wb') as f:
                    pickle.dump(t_by_video, f)

        self.gt_map = CrosstaskGroundTruth(self._K_by_task, t_by_video, self._release_root, self._task_names, self._remove_background)

        for task_name in self._task_names:
            if task_name not in self._videos_by_task:
                self._videos_by_task[task_name] = {}
            for video in self._video_names_by_task[task_name]:
                assert video not in self._videos_by_task[task_name]
                has_label = task_name in self.gt_map.gt_by_task

                nonbackground_timesteps = self.gt_map.nonbackground_timesteps_by_task[task_name][video] if (has_label and self._remove_background) else None
                self._videos_by_task[task_name][video] = CrosstaskVideo(
                    feature_root=self._feature_root,
                    dimensions_per_feature_group=self._dimensions_per_feature_group,
                    remove_background=self._remove_background,
                    nonbackground_timesteps=nonbackground_timesteps,
                    K=self._K_by_task[task_name],
                    gt=self.gt_map.gt_by_task[task_name][video] if has_label else None,
                    gt_with_background=self.gt_map.gt_with_background_by_task[task_name][video] if has_label else None,
                    name=video,
                    has_label=has_label,
                )

def extract_feature_groups(corpus):
    group_indices = {
        'i3d': (0, 1024),
        'resnet': (1024, 3072),
        'audio': (3072, 3200),

    }
    n_instances = len(corpus)
    grouped = defaultdict(dict)
    for idx in range(n_instances):
        video_name = corpus[idx]['video_name']
        features = corpus[idx]['features']
        for group, (start, end) in group_indices.items():
            grouped[group][video_name] = features[:,start:end]
    return grouped

def pca_and_serialize_features(release_root, raw_feature_root, output_feature_root, remove_background, pca_components_per_group=300, by_task=True, task_sets=None):
    if by_task:
        grouped_corpora = corpora_by_task(release_root, raw_feature_root, remove_background,
                                          full=True, split='all', task_sets=task_sets)
    else:
        grouped_corpora = {
            'all': CrosstaskCorpus(release_root, raw_feature_root, remove_background,
                                   split='all', task_sets=task_sets)
        }

    os.makedirs(output_feature_root, exist_ok=True)

    for corpora_group, corpus in grouped_corpora.items():
        logger.debug("saving features for task: {}".format(corpora_group))
        grouped_features = extract_feature_groups(corpus)
        transformed, pca_models = grouped_pca(grouped_features, pca_components_per_group, pca_models_by_group=None)
        for feature_group, vid_dict in transformed.items():
            logger.debug("\tsaving features for feature group: {}".format(feature_group))
            feature_group_dir = os.path.join(output_feature_root, feature_group)
            os.makedirs(feature_group_dir, exist_ok=True)
            for vid, features in vid_dict.items():
                fname = os.path.join(feature_group_dir, '{}.npy'.format(vid))
                np.save(fname, features)

if __name__ == "__main__":
    release_root = 'data/crosstask/crosstask_release'
    raw_feature_root = 'data/crosstask/crosstask_features'
    remove_background = False
    components = 200
    task_sets = ['primary']
    for by_task in [False, True]:
        output_feature_root = 'data/crosstask/crosstask_processed/crosstask_{}_pca-{}_{}_{}'.format(
            '+'.join(task_sets),
            components,
            'no-bkg' if remove_background else 'with-bkg',
            'by-task' if by_task else 'all-tasks',
        )

        pca_and_serialize_features(release_root, raw_feature_root,  output_feature_root, remove_background,
                                   pca_components_per_group=components, by_task=by_task, task_sets=task_sets)
