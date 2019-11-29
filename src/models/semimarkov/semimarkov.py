import copy
import numpy as np
import tqdm
import itertools

import torch
from data.corpus import Datasplit
from models.model import Model, make_optimizer, make_data_loader
from models.semimarkov.semimarkov_modules import SemiMarkovModule, ComponentSemiMarkovModule
from models.semimarkov import semimarkov_utils


class SemiMarkovModel(Model):
    @classmethod
    def add_args(cls, parser):
        SemiMarkovModule.add_args(parser)
        ComponentSemiMarkovModule.add_args(parser)
        parser.add_argument('--sm_component_model', action='store_true')

        parser.add_argument('--sm_constrain_transitions', action='store_true')

    @classmethod
    def from_args(cls, args, train_data):
        n_classes = train_data.corpus.n_classes
        feature_dim = train_data.feature_dim

        allow_self_transitions = True

        assert args.sm_max_span_length is not None
        if args.sm_constrain_transitions:
            assert args.task_specific_steps, "will get bad results with --sm_constrain_transitions if you don't also pass --task_specific_steps, because of multiple exits"
            # if not args.remove_background:
            #     raise NotImplementedError("--sm_constrain_transitions without --remove_background ")

            (
                allowed_starts, allowed_transitions, allowed_ends, ordered_indices_by_task
            ) = train_data.get_allowed_starts_and_transitions()
            if allow_self_transitions:
                for src in range(n_classes):
                    if src not in allowed_transitions:
                        allowed_transitions[src] = set()
                    allowed_transitions[src].add(src)
        else:
            allowed_starts, allowed_transitions, allowed_ends, ordered_indices_by_task = None, None, None, None

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
            model = ComponentSemiMarkovModule(
                args,
                n_classes,
                n_components=n_components,
                class_to_components=class_to_components,
                feature_dim=feature_dim,
                allow_self_transitions=allow_self_transitions,
                allowed_starts=allowed_starts,
                allowed_transitions=allowed_transitions,
                allowed_ends=allowed_ends,
            )
        else:
            model = SemiMarkovModule(
                args,
                n_classes,
                feature_dim,
                allow_self_transitions=allow_self_transitions,
                allowed_starts=allowed_starts,
                allowed_transitions=allowed_transitions,
                allowed_ends=allowed_ends,
            )
        return SemiMarkovModel(args, n_classes, feature_dim, model, ordered_indices_by_task)

    def __init__(self, args, n_classes, feature_dim, model, ordered_indices_by_task=None):
        self.args = args
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.model = model
        self.ordered_indices_by_task = ordered_indices_by_task
        if args.cuda:
            self.model.cuda()

    def fit_supervised(self, train_data: Datasplit):
        assert not self.args.sm_component_model
        assert not self.args.sm_constrain_transitions
        loader = make_data_loader(self.args, train_data, batch_by_task=False, shuffle=False, batch_size=1)
        features, labels = [], []
        for batch in loader:
            features.append(batch['features'].squeeze(0))
            labels.append(batch['gt_single'].squeeze(0))
        self.model.fit_supervised(features, labels)

    def make_additional_allowed_ends(self, tasks, lengths):
        if self.ordered_indices_by_task is not None:
            addl_allowed_ends = []
            for task, length in zip(tasks, lengths):
                ord_indices = self.ordered_indices_by_task[task]
                if length.item() < len(ord_indices):
                    this_allowed_ends = [ord_indices[length.item()-1]]
                else:
                    this_allowed_ends = []
                addl_allowed_ends.append(this_allowed_ends)
        else:
            addl_allowed_ends = None
        return addl_allowed_ends

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        self.model.train()
        self.model.flatten_parameters()
        if use_labels:
            assert not self.args.sm_constrain_transitions
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
                    spans = semimarkov_utils.labels_to_spans(labels, max_k=K)
                    use_mean_z = True
                else:
                    spans = None
                    use_mean_z = False

                addl_allowed_ends = self.make_additional_allowed_ends(tasks, lengths)

                nll = -self.model.log_likelihood(features,
                                                 lengths,
                                                 valid_classes_per_instance=task_indices,
                                                 spans=spans,
                                                 add_eos=True,
                                                 use_mean_z=use_mean_z,
                                                 additional_allowed_ends_per_instance=addl_allowed_ends)
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
                        param_norm = sum([p.norm()**2 for p in self.model.parameters()
                                          if p.requires_grad]).item()**0.5
                        gparam_norm = sum([p.grad.norm()**2 for p in self.model.parameters()
                                           if p.requires_grad and p.grad is not None]).item()**0.5
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
            callback_fn(epoch, {'train_loss': train_loss,
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
            tasks = batch['task_name']
            assert len(set(tasks)) == 1
            task = next(iter(tasks))

            addl_allowed_ends = self.make_additional_allowed_ends(tasks, lengths)

            # TODO: figure out under which eval conditions use_mean_z should be False
            pred_spans = self.model.viterbi(features, lengths, task_indices, add_eos=True, use_mean_z=True,
                                            additional_allowed_ends_per_instance=addl_allowed_ends)
            pred_labels = semimarkov_utils.spans_to_labels(pred_spans)

            # if self.args.sm_constrain_transitions:
            #     all_pred_span_indices = [
            #         [ix for ix, count in this_rle_spans]
            #         for this_rle_spans in semimarkov_utils.rle_spans(pred_spans, lengths)
            #     ]
            #     for i, indices in enumerate(all_pred_span_indices):
            #         remove_cons_dups = [ix for ix, group in itertools.groupby(indices)
            #                             if not ix in test_data.corpus._background_indices]
            #         non_bg_indices = [
            #             ix for ix in test_data.corpus.indices_by_task(task)
            #             if ix not in test_data.corpus._background_indices
            #         ]
            #         if len(remove_cons_dups) != len(non_bg_indices) and lengths[i].item() != len(remove_cons_dups):
            #             print("deduped: {}, indices: {}, length {}".format(
            #                 remove_cons_dups, non_bg_indices, lengths[i].item()
            #             ))
            #             # assert lengths[i].item() < len(non_bg_indices)

            pred_labels_trim_s = self.model.trim(pred_labels, lengths, check_eos=True)

            # assert len(pred_labels_trim_s) == 1, "batch size should be 1"
            for video, pred_labels_trim in zip(videos, pred_labels_trim_s):
                predictions[video] = pred_labels_trim.numpy()
                assert self.model.n_classes not in predictions[video], "predictions should not contain EOS: {}".format(
                    predictions[video])
        return predictions
