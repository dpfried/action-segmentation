import torch.optim
from torch.utils.data import DataLoader


def add_training_args(parser):
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--workers', type=int, default=1)


def make_optimizer(args, parameters):
    return torch.optim.Adam(parameters, lr=args.lr)


def make_data_loader(args, corpus, shuffle, batch_size=1):
    assert batch_size == 1, "other sizes not implemented"
    return DataLoader(
        corpus,
        batch_size=batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        drop_last=False,
        collate_fn=lambda batch: batch,
    )


class Model(object):
    def fit(self, train_data, callback_fn=None):
        raise NotImplementedError()

    def predict(self, test_data):
        raise NotImplementedError()
