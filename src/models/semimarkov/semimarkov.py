import copy
import numpy as np
import tqdm

import torch
from data.corpus import Datasplit
from models.model import Model, make_optimizer, make_data_loader
from models.semimarkov.semimarkov_modules import SemiMarkovModule, ComponentSemiMarkovModule


class SemiMarkovModel(Model):
    @classmethod
    def add_args(cls, parser):
        SemiMarkovModule.add_args(parser)
        ComponentSemiMarkovModule.add_args(parser)
        parser.add_argument('--sm_component_model', action='store_true')

    @classmethod
    def from_args(cls, args, train_data):
        n_classes = train_data.corpus.n_classes
        feature_dim = train_data.feature_dim

        assert args.sm_max_span_length is not None
        if args.sm_component_model:
            if args.sm_component_decompose_steps:
                # assert not args.task_specific_steps, "can't decompose steps unless steps are across tasks; you should remove --task_specific_steps"
                n_components = train_data.corpus.n_components
                class_to_components = copy.copy(train_data.corpus.label_indices2component_indices)
            else:
                n_components = n_classes
                class_to_components = {
                    cls: {cls}
                    for cls in range(n_classes)
                }
            model = ComponentSemiMarkovModule(args,
                                              n_classes,
                                              n_components=n_components,
                                              class_to_components=class_to_components,
                                              feature_dim=feature_dim,
                                              allow_self_transitions=True)
        else:
            model = SemiMarkovModule(args,
                                     n_classes,
                                     feature_dim,
                                     allow_self_transitions=True)
        return SemiMarkovModel(args, n_classes, feature_dim, model)

    def __init__(self, args, n_classes, feature_dim, model):
        self.args = args
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.model = model
        if args.cuda:
            self.model.cuda()

    def fit_supervised(self, train_data: Datasplit):
        assert not self.args.sm_component_model
        loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=False, batch_size=1)
        features, labels = [], []
        for batch in loader:
            features.append(batch['features'].squeeze(0))
            labels.append(batch['gt_single'].squeeze(0))
        self.model.fit_supervised(features, labels)

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        self.model.train()
        self.model.flatten_parameters()
        initialize = True
        if use_labels and self.args.sm_supervised_method in ['closed-form', 'closed-then-gradient']:
            self.fit_supervised(train_data)
            if self.args.sm_supervised_method == 'closed-then-gradient':
                initialize = False
                callback_fn(-1, {})
            else:
                return
        optimizer, scheduler = make_optimizer(self.args, self.model.parameters())
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
            # call here since we may set eval in callback_fn
            self.model.train()
            losses = []
            multi_batch_losses = []
            nlls = []
            kls = []
            num_frames = 0
            num_videos = 0
            train_nll = 0
            train_kl = 0
            for batch_ix, batch in enumerate(tqdm.tqdm(loader, ncols=80)):
                # if self.args.cuda:
                #     features = features.cuda()
                #     task_indices = task_indices.cuda()
                #     gt_single = gt_single.cuda()
                tasks = batch['task_name']
                videos = batch['video_name']
                features = batch['features']
                task_indices = batch['task_indices']
                lengths = batch['lengths']

                num_frames += lengths.sum().item()
                num_videos += len(lengths)

                # assert len( task_indices) == self.n_classes, "remove_background and multi-task fit() not implemented"

                if self.args.cuda:
                    features = features.cuda()
                    lengths = lengths.cuda()

                if use_labels:
                    labels = batch['gt_single']
                    if self.args.cuda:
                        labels = labels.cuda()
                    spans = SemiMarkovModule.labels_to_spans(labels, max_k=K)
                    use_mean_z = True
                else:
                    spans = None
                    use_mean_z = False

                nll = -self.model.log_likelihood(features,
                                                 lengths,
                                                 valid_classes_per_instance=task_indices,
                                                 spans=spans,
                                                 add_eos=True,
                                                 use_mean_z=use_mean_z)
                kl = self.model.kl.mean()
                if use_labels:
                    this_loss = nll
                else:
                    this_loss = nll + kl
                multi_batch_losses.append(this_loss)
                nlls.append(nll.item())
                kls.append(kl.item())

                train_nll += (nll.item() * len(videos))
                train_kl += (kl.item() * len(videos))

                losses.append(this_loss.item())

                if len(multi_batch_losses) >= self.args.batch_accumulation:
                    loss = sum(multi_batch_losses) / len(multi_batch_losses)
                    loss.backward()
                    multi_batch_losses = []

                    if self.args.print_every and (batch_ix % self.args.print_every == 0):
                        param_norm = sum([p.norm()**2 for p in self.model.parameters()]).item()**0.5
                        gparam_norm = sum([p.grad.norm()**2 for p in self.model.parameters()
                                           if p.grad is not None]).item()**0.5
                        log_str = 'Epoch: %02d, Batch: %03d/%03d, |Param|: %.6f, |GParam|: %.2f, lr: %.2E, ' + \
                                  'loss: %.4f, recon: %.4f, kl: %.4f, recon_bound: %.2f'
                        tqdm.tqdm.write(log_str %
                                        (epoch, batch_ix, len(loader), param_norm, gparam_norm,
                                         optimizer.param_groups[0]["lr"],
                                         (train_nll + train_kl) / num_videos,
                                         train_nll / num_frames,
                                         train_kl / num_frames,
                                         (train_nll + train_kl) / num_frames))
                    if self.args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    self.model.zero_grad()
            train_loss = np.mean(losses)
            if scheduler is not None:
                scheduler.step(train_loss)
            callback_fn(epoch, {'train_loss_vid_avg': train_loss,
                                'train_nll_frame_avg': train_nll / num_frames,
                                'train_kl_vid_avg': train_kl / num_videos,
                                'train_recon_bound': (train_kl + train_kl) / num_frames})

    def predict(self, test_data):
        self.model.eval()
        self.model.flatten_parameters()
        predictions = {}
        # for some reason, EOM errors happen more frequently in viterbi, so reduce batch size
        loader = make_data_loader(self.args, test_data, shuffle=False, batch_by_task=True, batch_size=self.args.batch_size // 2)
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
            # TODO: figure out under which eval conditions use_mean_z should be False
            pred_spans = self.model.viterbi(features, lengths, task_indices, add_eos=True, use_mean_z=True)
            pred_labels = SemiMarkovModule.spans_to_labels(pred_spans)
            pred_labels_trim_s = self.model.trim(pred_labels, lengths, check_eos=True)

            # assert len(pred_labels_trim_s) == 1, "batch size should be 1"
            for video, pred_labels_trim in zip(videos, pred_labels_trim_s):
                predictions[video] = pred_labels_trim.numpy()
                assert self.model.n_classes not in predictions[video], "predictions should not contain EOS: {}".format(
                    predictions[video])
        return predictions
