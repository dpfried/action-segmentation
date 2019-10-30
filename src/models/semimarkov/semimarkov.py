import numpy as np
import tqdm

from data.corpus import Datasplit
from models.model import Model, make_optimizer, make_data_loader
from models.semimarkov.semimarkov_modules import SemiMarkovModule, ComponentSemiMarkovModule


class SemiMarkovModel(Model):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--sm_max_span_length', type=int)
        parser.add_argument('--sm_supervised_state_smoothing', type=float, default=1e-2)
        parser.add_argument('--sm_supervised_length_smoothing', type=float, default=1e-1)
        parser.add_argument('--sm_supervised_method',
                            choices=['closed-form', 'gradient-based', 'closed-then-gradient'],
                            default='closed-form')

        parser.add_argument('--sm_component_model', action='store_true')

    @classmethod
    def from_args(cls, args, train_data):
        n_classes = train_data.corpus.n_classes
        feature_dim = train_data.feature_dim
        return SemiMarkovModel(args, n_classes, feature_dim)

    def __init__(self, args, n_classes, feature_dim):
        self.args = args
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        assert self.args.sm_max_span_length is not None
        if args.sm_component_model:
            n_components = self.n_classes
            class_to_components = {
                cls: {cls}
                for cls in range(self.n_classes)
            }
            self.model = ComponentSemiMarkovModule(self.n_classes,
                                                   n_components=n_components,
                                                   class_to_components=class_to_components,
                                                   feature_dim=self.feature_dim,
                                                   embedding_dim=100,
                                                   allow_self_transitions=True,
                                                   max_k=self.args.sm_max_span_length,
                                                   per_class_bias=True)
        else:
            self.model = SemiMarkovModule(self.n_classes,
                                          self.feature_dim,
                                          max_k=self.args.sm_max_span_length,
                                          allow_self_transitions=True)
        if args.cuda:
            self.model.cuda()

    def fit_supervised(self, train_data: Datasplit):
        loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=False, batch_size=1)
        features, labels = [], []
        for batch in loader:
            features.append(batch['features'].squeeze(0))
            labels.append(batch['gt_single'].squeeze(0))
        self.model.fit_supervised(features, labels, self.args.sm_supervised_state_smoothing,
                                  self.args.sm_supervised_length_smoothing)

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        self.model.train()
        initialize = True
        if use_labels and self.args.sm_supervised_method in ['closed-form', 'closed-then-gradient']:
            self.fit_supervised(train_data)
            if self.args.sm_supervised_method == 'closed-then-gradient':
                initialize = False
                callback_fn(-1, {})
            else:
                return
        optimizer = make_optimizer(self.args, self.model.parameters())
        big_loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=True, batch_size=100)
        samp = next(iter(big_loader))
        big_features = samp['features']
        big_lengths = samp['lengths']
        if self.args.cuda:
            big_features = big_features.cuda()
            big_lengths = big_lengths.cuda()

        if initialize:
            self.model.initialize_gaussian(big_features, big_lengths)

        loader = make_data_loader(self.args, train_data, batch_by_task=True, shuffle=True, batch_size=self.args.batch_size)

        # all_features = [sample['features'] for batch in loader for sample in batch]
        # if self.args.cuda:
        #     all_features = [feats.cuda() for feats in all_features]

        C = self.n_classes
        K = self.args.sm_max_span_length

        for epoch in range(self.args.epochs):
            losses = []
            for batch in tqdm.tqdm(loader, ncols=80):
                # if self.args.cuda:
                #     features = features.cuda()
                #     task_indices = task_indices.cuda()
                #     gt_single = gt_single.cuda()
                tasks = batch['task_name']
                videos = batch['video_name']
                features = batch['features']
                labels = batch['gt_single']
                task_indices = batch['task_indices']
                lengths = batch['lengths']

                # assert len( task_indices) == self.n_classes, "remove_background and multi-task fit() not implemented"

                # # add a batch dimension
                # lengths = torch.LongTensor([features.size(0)]).unsqueeze(0)
                # features = features.unsqueeze(0)
                # labels = labels.unsqueeze(0)
                # task_indices = task_indices.unsqueeze(0)

                if self.args.cuda:
                    features = features.cuda()
                    lengths = lengths.cuda()
                    labels = labels.cuda()

                if use_labels:
                    spans = SemiMarkovModule.labels_to_spans(labels, max_k=K)
                else:
                    spans = None

                this_loss = -self.model.log_likelihood(features,
                                                       lengths,
                                                       valid_classes_per_instance=task_indices,
                                                       spans=spans,
                                                       add_eos=True)
                this_loss.backward()

                losses.append(this_loss.item())
                # TODO: grad clipping?
                optimizer.step()
                self.model.zero_grad()
            callback_fn(epoch, {'train_loss': np.mean(losses)})

    def predict(self, test_data):
        self.model.eval()
        predictions = {}
        loader = make_data_loader(self.args, test_data, shuffle=False, batch_by_task=True, batch_size=self.args.batch_size)
        for batch in tqdm.tqdm(loader, ncols=80):
            features = batch['features']
            task_indices = batch['task_indices']
            lengths = batch['lengths']

            # add a batch dimension
            # lengths = torch.LongTensor([features.size(0)]).unsqueeze(0)
            # features = features.unsqueeze(0)
            # task_indices = task_indices.unsqueeze(0)

            if self.args.cuda:
                features = features.cuda()
                task_indices = [ti.cuda() for ti in task_indices]
                lengths = lengths.cuda()

            videos = batch['video_name']

            pred_spans = self.model.viterbi(features, lengths, task_indices, add_eos=True)
            pred_labels = SemiMarkovModule.spans_to_labels(pred_spans)
            pred_labels_trim_s = self.model.trim(pred_labels, lengths, check_eos=True)

            # assert len(pred_labels_trim_s) == 1, "batch size should be 1"
            for video, pred_labels_trim in zip(videos, pred_labels_trim_s):
                predictions[video] = pred_labels_trim.numpy()
                assert self.model.n_classes not in predictions[video], "predictions should not contain EOS: {}".format(
                    predictions[video])
        return predictions
