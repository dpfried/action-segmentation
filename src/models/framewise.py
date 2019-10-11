import tqdm
import numpy as np
import torch
import torch.nn as nn
from models.model import Model, make_optimizer, make_data_loader
from sklearn.mixture import GaussianMixture

from models.semimarkov import semimarkov_sufficient_stats

from data.corpus import Datasplit


class FeedForward(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--ff_dropout_p', type=float, default=0.1)
        parser.add_argument('--ff_hidden_layers', type=int, default=0)
        parser.add_argument('--ff_hidden_dim', type=int, default=200)

    def __init__(self, args, input_dim, output_dim):
        super(FeedForward, self).__init__()
        self.args = args
        layers = [nn.Dropout(p=args.ff_dropout_p)]
        layers.append(nn.Linear(input_dim,output_dim if args.ff_hidden_layers == 0 else args.ff_hidden_dim))
        if args.ff_hidden_layers > 0:
            for l_ix in range(args.ff_hidden_layers):
                # TODO: consider adding dropout in here
                layers.append(nn.ReLU())
                layers.append(nn.Linear(args.ff_hidden_dim, args.ff_hidden_dim if l_ix < args.ff_hidden_layers - 1 else output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x, valid_classes=None):
        batch_size = x.size(0)
        logits = self.layers(x)
        if valid_classes is not None:
            mask = torch.full_like(logits, -float("inf"))
            mask[:,valid_classes] = 0
            logits = logits + mask
        return logits


class FramewiseDiscriminative(Model):
    @classmethod
    def add_args(cls, parser):
        FeedForward.add_args(parser)

    @classmethod
    def from_args(cls, args, train_data: Datasplit):
        return FramewiseDiscriminative(args, train_data)

    def __init__(self, args, train_data: Datasplit):
        self.args = args
        #self.n_classes = sum(len(indices) for indices in train_data.groundtruth.indices_by_task.values())
        self.n_classes = train_data._corpus.n_classes
        self.model = FeedForward(args,
                                 input_dim=train_data.feature_dim,
                                 output_dim=self.n_classes)
        if args.cuda:
            self.model.cuda()

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        assert use_labels
        loss = nn.CrossEntropyLoss()
        self.model.train()
        optimizer = make_optimizer(self.args, self.model.parameters())
        loader = make_data_loader(self.args, train_data, shuffle=True, batch_size=1)

        for epoch in range(self.args.epochs):
            losses = []
            # for batch in tqdm.tqdm(loader, ncols=80):
            for batch in loader:
                for sample in batch:
                    task = sample['task_name']
                    video = sample['video_name']
                    features = sample['features']
                    gt_single = sample['gt_single']
                    task_indices = sample['task_indices']
                    if self.args.cuda:
                        features = features.cuda()
                        task_indices = task_indices.cuda()
                        gt_single = gt_single.cuda()
                    logits = self.model.forward(features, valid_classes=task_indices)

                    this_loss = loss(logits, gt_single)
                    losses.append(this_loss.item())
                    this_loss.backward()

                    optimizer.step()
                    self.model.zero_grad()
            callback_fn(epoch, {'train_loss': np.mean(losses)})
            # if evaluate_on_data_fn is not None:
            #     train_mof = evaluate_on_data_fn(self, train_data, 'train')
            #     dev_mof = evaluate_on_data_fn(self, dev_data, 'dev')
            #     dev_mof_by_epoch[epoch] = dev_mof
            #     log_str += ("\ttrain mof: {:.4f}".format(train_mof))
            #     log_str += ("\tdev mof: {:.4f}".format(dev_mof))

    def predict(self, test_data: Datasplit):
        self.model.eval()
        predictions = {}
        loader = make_data_loader(self.args, test_data, shuffle=False, batch_size=1)
        for batch in loader:
            for sample in batch:
                features = sample['features']
                task_indices = sample['task_indices']
                if self.args.cuda:
                    features = features.cuda()
                    task_indices = task_indices.cuda()
                video = sample['video_name']
                logits = self.model.forward(features, valid_classes=task_indices)
                preds = logits.max(dim=1)[1]
                # handle the edge case where there's only a single instance, in which case preds.size() <= 1
                if len(preds.size()) > 1:
                    preds = preds.squeeze(-1)
                predictions[video] = preds.detach().cpu().numpy()
        return predictions


class FramewiseGaussianMixture(Model):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--gm_covariance', choices=['full', 'diag', 'tied', 'tied_diag'], default='tied_diag')

    @classmethod
    def from_args(cls, args, train_data):
        n_classes = train_data._corpus.n_classes
        feature_dim = train_data.feature_dim
        return FramewiseGaussianMixture(args, n_classes, feature_dim)

    def __init__(self, args, n_classes, feature_dim):
        self.args = args
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.model = None

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        assert use_labels
        gmm, stats = semimarkov_sufficient_stats(train_data, self.args.gm_covariance, self.n_classes)
        self.model = gmm

    def predict(self, test_data):
        assert self.model is not None
        predictions = {}
        # for i in tqdm.tqdm(list(range(len(test_data))), ncols=80):
        for i in range(len(test_data)):
            sample = test_data.__getitem__(i, wrap_torch=False)
            X = sample['features']
            mask_indices = list(set(range(self.n_classes)) - set(sample['task_indices']))
            if mask_indices:
                probs = self.model.predict_proba(X)
                probs[:, mask_indices] = 0
                probs /= probs.sum(axis=1)[:,None]
                preds = probs.argmax(axis=1)
            else:
                preds = self.model.predict(X)
            predictions[sample['video_name']] = preds
        return predictions
