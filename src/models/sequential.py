import tqdm
import numpy as np
import torch
import torch.nn as nn
from models.model import Model, make_optimizer, make_data_loader
from utils.utils import all_equal

from models.semimarkov.semimarkov_modules import semimarkov_sufficient_stats

from collections import Counter

from data.corpus import Datasplit


class Encoder(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--seq_num_layers', type=int, default=2)

    def __init__(self, args, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.args = args
        assert output_dim % 2 == 0
        # TODO: dropout?
        self.encoder = nn.LSTM(input_dim, output_dim // 2, bidirectional=True, num_layers=args.seq_num_layers, batch_first=True)

    def forward(self, features, lengths, output_padding_value=0):
        batch_size = features.size(0)

        packed = nn.utils.rnn.pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)

        encoded_packed, _ = self.encoder(packed)

        encoded, lengths2 = nn.utils.rnn.pad_packed_sequence(encoded_packed, batch_first=True, padding_value=output_padding_value)

        return encoded

class SequentialPredictFrames(nn.Module):
    @classmethod
    def add_args(cls, parser):
        Encoder.add_args(parser)
        parser.add_argument('--seq_hidden_size', type=int, default=200)

    def __init__(self, args, input_dim, num_classes):
        super(SequentialPredictFrames, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.encoder = Encoder(self.args, input_dim, args.seq_hidden_size)
        self.proj = nn.Linear(args.seq_hidden_size, num_classes)

    def forward(self, features, lengths, valid_classes_per_instance=None):
        # batch_size x max_len x seq_hidden_size
        encoded = self.encoder(features, lengths, output_padding_value=0)
        # batch_size x max_len x num_classes
        logits = self.proj(encoded)
        if valid_classes_per_instance is not None:
            assert all_equal(set(vc.detach().cpu().numpy()) for vc in
                             valid_classes_per_instance), "must have same valid_classes for all instances in the batch"
            valid_classes = valid_classes_per_instance[0]
            mask = torch.full_like(logits, -float("inf"))
            mask[:,:,valid_classes] = 0
            logits = logits + mask
        return logits

class SequentialDiscriminative(Model):
    @classmethod
    def add_args(cls, parser):
        SequentialPredictFrames.add_args(parser)

    @classmethod
    def from_args(cls, args, train_data: Datasplit):
        return SequentialDiscriminative(args, train_data)

    def __init__(self, args, train_data: Datasplit):
        self.args = args
        #self.n_classes = sum(len(indices) for indices in train_data.groundtruth.indices_by_task.values())
        self.n_classes = train_data._corpus.n_classes
        self.model = SequentialPredictFrames(args, input_dim=train_data.feature_dim, num_classes=self.n_classes)
        if args.cuda:
            self.model.cuda()

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        assert use_labels
        IGNORE = -100
        loss = nn.CrossEntropyLoss(ignore_index=IGNORE)
        optimizer, scheduler = make_optimizer(self.args, self.model.parameters())
        loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=True, batch_size=self.args.batch_size)

        for epoch in range(self.args.epochs):
            # call here since we may set eval in callback_fn
            self.model.train()
            losses = []
            assert self.args.batch_accumulation <= 1
            for batch in tqdm.tqdm(loader, ncols=80):
            # for batch in loader:
                tasks = batch['task_name']
                videos = batch['video_name']
                features = batch['features']
                gt_single = batch['gt_single']
                task_indices = batch['task_indices']
                max_len = features.size(1)
                lengths = batch['lengths']
                invalid_mask = torch.arange(max_len).expand(len(lengths), max_len) >= lengths.unsqueeze(1)
                if self.args.cuda:
                    features = features.cuda()
                    lengths = lengths.cuda()
                    task_indices = [indx.cuda() for indx in task_indices]
                    gt_single = gt_single.cuda()
                    invalid_mask = invalid_mask.cuda()
                gt_single.masked_fill_(invalid_mask, IGNORE)
                # batch_size x max_len x num_classes
                logits = self.model(features, lengths, valid_classes_per_instance=task_indices)

                this_loss = loss(logits.view(-1, logits.size(-1)), gt_single.flatten())
                losses.append(this_loss.item())
                this_loss.backward()

                optimizer.step()
                self.model.zero_grad()
            train_loss = np.mean(losses)
            if scheduler is not None:
                scheduler.step(train_loss)
            callback_fn(epoch, {'train_loss': train_loss})

            # if evaluate_on_data_fn is not None:
            #     train_mof = evaluate_on_data_fn(self, train_data, 'train')
            #     dev_mof = evaluate_on_data_fn(self, dev_data, 'dev')
            #     dev_mof_by_epoch[epoch] = dev_mof
            #     log_str += ("\ttrain mof: {:.4f}".format(train_mof))
            #     log_str += ("\tdev mof: {:.4f}".format(dev_mof))

    def predict(self, test_data: Datasplit):
        self.model.eval()
        predictions = {}
        loader = make_data_loader(self.args, test_data, batch_by_task=False, shuffle=False, batch_size=1)
        for batch in loader:
            features = batch['features']
            lengths = batch['lengths']
            task_indices = batch['task_indices']
            if self.args.cuda:
                features = features.cuda()
                lengths = lengths.cuda()
                task_indices = [indx.cuda() for indx in task_indices]
            videos = batch['video_name']
            assert all_equal(videos)
            video = next(iter(videos))
            # batch_size x length x num_classes
            with torch.no_grad():
                logits = self.model(features, lengths, valid_classes_per_instance=task_indices)
                preds = logits.max(dim=-1)[1]
                preds = preds.squeeze(0)
                assert preds.ndim == 1
                predictions[video] = preds.detach().cpu().numpy()
        return predictions

