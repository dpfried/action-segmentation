import torch.optim
from torch.utils.data import DataLoader

from data.corpus import Datasplit

from utils.utils import all_equal


def add_training_args(parser):
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=1)


def make_optimizer(args, parameters):
    return torch.optim.Adam(parameters, lr=args.lr)


def padding_colate(data_samples):
    unpacked = {
        key: [samp[key] for samp in data_samples]
        for key in next(iter(data_samples)).keys()
    }

    lengths = [feats.size(0) for feats in unpacked['features']]
    # batch_size = len(lengths)
    # max_length = max(lengths)
    # lengths_t = torch.LongTensor(lengths)

    pad_keys = ['gt_single', 'features']
    nopad_keys = ['task_name', 'video_name', 'task_indices', 'gt', 'gt_with_background']
    data = {k: v for k, v in unpacked.items() if k in nopad_keys}
    data['lengths'] = torch.LongTensor(lengths)

    for key in pad_keys:
        data[key] = torch.nn.utils.rnn.pad_sequence(unpacked[key], batch_first=True, padding_value=0)

    return data


def make_data_loader(args, datasplit: Datasplit, shuffle, batch_by_task, batch_size=1):
    # assert batch_size == 1, "other sizes not implemented"
    return DataLoader(
        datasplit,
        # batch_size=batch_size,
        num_workers=args.workers,
        # shuffle=shuffle,
        # drop_last=False,
        # collate_fn=lambda batch: batch,
        collate_fn=padding_colate,
        batch_sampler=datasplit.batch_sampler(batch_size, batch_by_task, shuffle)
    )


class Model(object):
    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        raise NotImplementedError()

    def predict(self, test_data):
        raise NotImplementedError()
