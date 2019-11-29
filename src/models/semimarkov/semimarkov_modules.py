from typing import Dict, Set

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.distributions import MultivariateNormal, Poisson
from torch.nn.init import xavier_uniform_
from torch_struct import SemiMarkovCRF

from models.framewise import FeedForward
from models.sequential import Encoder
from models.flow import NICETrans
from utils.utils import all_equal

from models.semimarkov.semimarkov_utils import semimarkov_sufficient_stats

BIG_NEG = -1e9
# BIG_NEG = -1e30

# share all params between annotated background steps ?
DEBUG_ANNOTATE = False

def sliding_sum(inputs, k):
    # inputs: b x T x c
    # sums sliding windows along the T dim, of length k
    batch_size = inputs.size(0)
    assert k > 0
    if k == 1:
        return inputs
    sliding_windows = F.unfold(inputs.unsqueeze(1),
                               kernel_size=(k, 1),
                               padding=(k, 0)).reshape(batch_size, k, -1, inputs.size(-1))
    sliding_summed = sliding_windows.sum(dim=1)
    ret = sliding_summed[:, k:-1, :]
    assert ret.shape == inputs.shape
    return ret


class ResidualLayer(nn.Module):
    def __init__(self, in_dim=100, out_dim=100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x


class SemiMarkovModule(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--sm_max_span_length', type=int, default=20)
        parser.add_argument('--sm_supervised_state_smoothing', type=float, default=1e-2)
        parser.add_argument('--sm_supervised_length_smoothing', type=float, default=1e-1)
        parser.add_argument('--sm_supervised_method',
                            choices=['closed-form', 'gradient-based', 'closed-then-gradient'],
                            default='closed-form')

        # parser.add_argument('--sm_projection_dim', type=int)
        parser.add_argument('--sm_feature_projection', action='store_true', help='use a flow')
        NICETrans.add_args(parser)

    def __init__(self, args, n_classes, n_dims,
                 allow_self_transitions=False,
                 allowed_starts: Set[int] = None,
                 allowed_transitions: Dict[int, Set[int]] = None,
                 allowed_ends: Set[int] = None,
                 merge_classes: Dict[int, int] = None,
                 ):
        super(SemiMarkovModule, self).__init__()
        self.args = args
        self.n_classes = n_classes
        self.input_feature_dim = n_dims
        # if self.args.sm_feature_projection:
        #     self.feature_dim = self.args.sm_projection_dim or n_dims
        # else:
        #     self.feature_dim = n_dims
        self.feature_dim = n_dims
        self.allow_self_transitions = allow_self_transitions
        self.init_params()
        self.init_projector()
        if allowed_starts is not None:
            assert allowed_transitions is not None
            self.set_transition_constraints(allowed_starts, allowed_transitions, allowed_ends)
        else:
            self.remove_transition_constraints()
        self.max_k = args.sm_max_span_length
        # self._learn_transitions = learn_transitions

        self.merge_classes = merge_classes
        # if self.merge_classes is not None:
        #     self.inv_merge_classes = {}
        #     for sink, src in merge_classes.items():
        #         assert src not in self.inv_merge_classes
        #         self.inv_merge_classes[src] = sink
        # else:
        #     self.merge_classes = None

    # @property
    # def learn_transitions(self):
    #     # backward compat for unpickling existing models
    #     try:
    #         return self._learn_transitions
    #     except:
    #         self._learn_transitions = True
    #         return self._learn_transitions

    def init_projector(self):
        if self.args.sm_feature_projection:
            # self.feature_projector = nn.Sequential(
            #     FeedForward(self.args,
            #                 self.input_feature_dim,
            #                 self.feature_dim),
            # )
            self.feature_projector = NICETrans(self.args, features=self.feature_dim)
        else:
            self.feature_projector = None

    def init_params(self):
        poisson_log_rates = torch.zeros(self.n_classes).float()
        self.poisson_log_rates = nn.Parameter(poisson_log_rates, requires_grad=True)

        gaussian_means = torch.zeros(self.n_classes, self.feature_dim).float()
        self.gaussian_means = nn.Parameter(gaussian_means, requires_grad=True)

        # shared, tied, diagonal covariance matrix
        gaussian_cov = torch.eye(self.feature_dim).float()
        self.gaussian_cov = nn.Parameter(gaussian_cov, requires_grad=False)

        # target x source
        transition_logits = torch.zeros(self.n_classes, self.n_classes).float()
        self.transition_logits = nn.Parameter(transition_logits, requires_grad=True)

        init_logits = torch.zeros(self.n_classes).float()
        self.init_logits = nn.Parameter(init_logits, requires_grad=True)
        torch.nn.init.uniform_(self.init_logits, 0, 1)

    def flatten_parameters(self):
        pass

    def remove_transition_constraints(self):
        self.transition_constraints = None
        self.init_constraints = None
        self.allowed_ends = None

    def set_transition_constraints(
            self,
            allowed_starts: Set[int],
            allowed_transitions: Dict[int, Set[int]],
            allowed_ends: Set[int]
    ):
        # make it a parameter so that it can get moved to cuda
        self.init_constraints = nn.Parameter(
            torch.full((self.n_classes,), 1, dtype=torch.bool),
            requires_grad=False
        )
        assert all(x >= 0 for x in allowed_starts)
        self.init_constraints[torch.LongTensor(list(sorted(allowed_starts)))] = 0

        # make it a parameter so that it can get moved to cuda
        self.transition_constraints = nn.Parameter(
            torch.full((self.n_classes, self.n_classes), 1, dtype=torch.bool),
            requires_grad=False
        )
        # TODO: consider vectorizing this. but it's probably fine if we only call once and allowed_transitions is sparse
        for src, targets in allowed_transitions.items():
            for tgt in targets:
                self.transition_constraints[tgt,src] = 0

        self.allowed_ends = allowed_ends

    def fit_supervised(self, feature_list, label_list):
        if self.feature_projector is not None:
            raise NotImplementedError("fit_supervised closed form with feature projector")

        if self.transition_constraints is not None or self.init_constraints is not None:
            raise NotImplementedError("fit_supervised closed form with constrained state transitions")

        emission_gmm, stats = semimarkov_sufficient_stats(
            feature_list, label_list,
            covariance_type='tied_diag',
            n_classes=self.n_classes,
            max_k=self.max_k,
        )
        if self.merge_classes is not None:
            label_list_merged = [
                torch.LongTensor([self.merge_classes[ix.item()] for ix in labels]).to(labels.device)
                for labels in label_list
            ]
            emission_gmm_merged, stats_merged = semimarkov_sufficient_stats(
                feature_list, label_list_merged,
                covariance_type='tied_diag',
                n_classes=self.n_classes,
                max_k=self.max_k,
            )
        else:
            emission_gmm_merged, stats_merged = emission_gmm, stats


        if DEBUG_ANNOTATE:
            emission_gmm = emission_gmm_merged
            stats = stats_merged

        # transition probs use unmerged classes
        init_probs = (stats['span_start_counts'] + self.args.sm_supervised_state_smoothing) / float(
            stats['instance_count'] + self.args.sm_supervised_state_smoothing * self.n_classes)
        init_probs[np.isnan(init_probs)] = 0
        # assert np.allclose(init_probs.sum(), 1.0), init_probs
        self.init_logits.data.zero_()
        self.init_logits.data.add_(torch.from_numpy(init_probs).to(device=self.init_logits.device).log())

        smoothed_trans_counts = stats['span_transition_counts'] + self.args.sm_supervised_state_smoothing

        trans_probs = smoothed_trans_counts / smoothed_trans_counts.sum(axis=0)[None, :]
        trans_probs[np.isnan(trans_probs)] = 0
        # to, from -- so rows should sum to 1
        # assert np.allclose(trans_probs.sum(axis=0), 1.0, rtol=1e-3), (trans_probs.sum(axis=0), trans_probs)
        self.transition_logits.data.zero_()
        self.transition_logits.data.add_(torch.from_numpy(trans_probs).to(device=self.transition_logits.device).log())

        # lengths and emissions use merged classes
        mean_lengths = (stats_merged['span_lengths'] + self.args.sm_supervised_length_smoothing) / (
                stats_merged['span_counts'] + self.args.sm_supervised_length_smoothing)
        self.poisson_log_rates.data.zero_()
        self.poisson_log_rates.data.add_(torch.from_numpy(mean_lengths).to(device=self.poisson_log_rates.device).log())

        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(
            torch.from_numpy(emission_gmm_merged.means_).to(device=self.gaussian_means.device).float())

        self.gaussian_cov.data.zero_()
        self.gaussian_cov.data.add_(
            torch.diag(torch.from_numpy(emission_gmm_merged.covariances_[0]).to(device=self.gaussian_cov.device).float()))

    def _initialize_gaussian_means(self, mean):
        # self.gaussian_means.data = mean.expand((self.n_classes, self.n_dims))
        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(mean.expand((self.n_classes, self.feature_dim)))

    def initialize_gaussian_from_feature_list(self, features):
        feats = torch.cat(features, dim=0)
        if self.feature_projector:
            feats, jacobian = self.feature_projector(feats)
        assert feats.dim() == 2
        n_dim = feats.size(1)
        assert n_dim == self.feature_dim
        mean = feats.mean(dim=0, keepdim=True)
        self._initialize_gaussian_means(mean)
        #
        # TODO: consider using the biased estimator, with torch >= 1.2?
        self.gaussian_cov.data = torch.diag(feats.var(dim=0))

    def initialize_gaussian(self, data, lengths):
        batch_size, N, n_dim = data.size()
        assert lengths.size(0) == batch_size
        feats = []
        for i in range(batch_size):
            feats.append(data[i, :lengths[i]])
        self.initialize_gaussian_from_feature_list(feats)

    def initial_log_probs(self, valid_classes):
        logits = self.init_logits
        if self.init_constraints is not None:
            logits = logits.masked_fill(self.init_constraints, BIG_NEG)

        if DEBUG_ANNOTATE and self.merge_classes is not None and valid_classes is not None:
            valid_classes = torch.LongTensor(
                [self.merge_classes[ix.item()] for ix in valid_classes],
            ).to(logits.device)

        if valid_classes is not None:
            logits = logits[valid_classes]
        return F.log_softmax(logits, dim=0)

    def transition_log_probs(self, valid_classes):
        transition_logits = self.transition_logits
        if self.transition_constraints is not None:
            transition_logits = transition_logits.masked_fill(self.transition_constraints, BIG_NEG)

        if DEBUG_ANNOTATE and self.merge_classes is not None and valid_classes is not None:
            valid_classes = torch.LongTensor(
                [self.merge_classes[ix.item()] for ix in valid_classes],
            ).to(transition_logits.device)

        if valid_classes is not None:
            transition_logits = transition_logits[valid_classes][:, valid_classes]
            n_classes = len(valid_classes)
        else:
            n_classes = self.n_classes
        if self.allow_self_transitions:
            masked = transition_logits
        else:
            masked = transition_logits.masked_fill(
                torch.eye(n_classes, device=self.transition_logits.device).bool(),
                BIG_NEG
            )
        # transition_logits are indexed: to_state, from_state
        # so each column should be normalized (in log-space)
        return F.log_softmax(masked, dim=0)

    def _emission_log_probs_with_means(self, features, class_means):
        # num_classes, d_ = class_means.size()
        # b, N, d = features.size()
        # assert d == d_, (d, d_)
        # feats_reshaped = features.reshape(-1, d)
        # dists = [
        #     MultivariateNormal(loc=mean,
        #                        covariance_matrix=self.gaussian_cov)
        #     for mean in class_means
        # ]
        # log_probs = [
        #     dist.log_prob(feats_reshaped).reshape(b, N, 1)  # b x
        #     for dist in dists
        # ]
        # return torch.cat(log_probs, dim=2)
        B, N, d = features.size()
        if class_means.dim() == 2:
            num_classes, d_ = class_means.size()
            assert d == d_, (d, d_)
            class_means = class_means.unsqueeze(0)
        else:
            _, num_classes, d_ = class_means.size()
            assert d == d_, (d, d_)
        class_means = class_means.expand(B, num_classes, d)

        scale_tril = self.gaussian_cov.sqrt()

        # b x N x num_classes

        log_probs = []
        for c in range(num_classes):
            # b x d
            this_means = class_means[:, c, :]
            dist = MultivariateNormal(loc=this_means, scale_tril=scale_tril)
            #  features.transpose(0,1): N x b x d
            log_probs.append(
                dist.log_prob(features.transpose(0, 1)).transpose(0, 1).unsqueeze(-1)  # b x N x 1
            )
        return torch.cat(log_probs, dim=2)

    def emission_log_probs(self, features, valid_classes):
        if valid_classes is None:
            class_indices = torch.LongTensor(
                list(range(self.n_classes)),
            ).to(self.gaussian_means.device)
        else:
            class_indices = valid_classes
        # if self.inv_merge_classes is not None:
        #     class_indices = [self.inv_merge_classes[ix] for ix in class_indices]
        if self.merge_classes is not None:
            class_indices = torch.LongTensor(
                [self.merge_classes[ix.item()] for ix in class_indices],
            ).to(self.gaussian_means.device)
        class_means = self.gaussian_means[class_indices]
        return self._emission_log_probs_with_means(features, class_means)

    def _length_log_probs_with_rates(self, log_rates):
        n_classes = log_rates.size(-1)
        max_length = self.max_k
        # max_length x n_classes
        time_steps = torch.arange(max_length, device=log_rates.device).unsqueeze(-1).expand(max_length,
                                                                                            n_classes).float()
        poissons = Poisson(torch.exp(log_rates))
        if log_rates.dim() == 2:
            time_steps = time_steps.unsqueeze(1).expand(max_length, log_rates.size(0), n_classes)
            return poissons.log_prob(time_steps).transpose(0, 1)
        else:
            assert log_rates.dim() == 1
            return poissons.log_prob(time_steps)

    def length_log_probs(self, valid_classes):
        if valid_classes is None:
            class_indices = torch.LongTensor(list(range(self.n_classes))).to(self.poisson_log_rates.device)
            n_classes = self.n_classes
        else:
            class_indices = valid_classes
            n_classes = len(valid_classes)
        # if self.inv_merge_classes is not None:
        #     class_indices = [self.inv_merge_classes[ix] for ix in class_indices]
        if self.merge_classes is not None:
            class_indices = torch.LongTensor(
                [self.merge_classes[ix.item()] for ix in class_indices],
            ).to(self.poisson_log_rates.device)
        log_rates = self.poisson_log_rates[class_indices]
        return self._length_log_probs_with_rates(log_rates)

    @staticmethod
    def log_hsmm(transition, emission_scores, init, length_scores, lengths, add_eos,
                 all_batched=False, allowed_ends_per_instance=None):
        """
        Convert HSMM to a linear chain.
        Parameters (if all_batched = True):
            transition: C X C
            emission_scores: b x N x C
            init: C
            length_scores: K x C
            add_eos: bool, whether to augment with an EOS class (with index C) which can only appear in the final timestep
        OR, if all_batched = False:
            transition: b x C X C
            emission_scores: b x N x C
            init: b x C
            length_scores: b x K x C
            add_eos: bool, whether to augment with an EOS class (with index C) which can only appear in the final timestep

            all_batched: if False, emission_scores is the only tensor with a batch dimension
        Returns:
            edges: b x (N-1) x C x C if not add_eos, or b x (N) x (C+1) x (C+1) if add_eos
        """
        b, N_1, C_1 = emission_scores.shape
        if all_batched:
            _, K, _C = length_scores.shape
            assert C_1 == _C
        else:
            K, _C = length_scores.shape
            assert C_1 == _C
            transition = transition.unsqueeze(0).expand(b, C_1, C_1)
            length_scores = length_scores.unsqueeze(0).expand(b, K, C_1)
            init = init.unsqueeze(0).expand(b, C_1)
            # emission_scores is already batched

        if K > N_1:
            K = N_1
            length_scores = length_scores[:, :K]
        # assert N_1 >= K
        # need to add EOS token
        if add_eos:
            N = N_1 + 1
            C = C_1 + 1
        else:
            N = N_1
            C = C_1
        if add_eos:
            transition_augmented = torch.full((b, C, C), BIG_NEG, device=transition.device)
            transition_augmented[:, :C_1, :C_1] = transition
            if allowed_ends_per_instance is None:
                # can transition from anything to EOS
                transition_augmented[:, C_1, :] = 0
            else:
                # can transition from any of allowed_ends to EOS
                for i, allowed_ends in enumerate(allowed_ends_per_instance):
                    assert len(allowed_ends) > 0
                    transition_augmented[i, C_1, allowed_ends] = 0

            init_augmented = torch.full((b, C), BIG_NEG, device=init.device)
            init_augmented[:, :C_1] = init

            length_scores_augmented = torch.full((b, K, C), BIG_NEG, device=length_scores.device)
            length_scores_augmented[:, :, :C_1] = length_scores
            # EOS must be length 1, although I don't think this is checked in the dp
            length_scores_augmented[:, 1, C_1] = 0

            emission_augmented = torch.full((b, N, C), BIG_NEG, device=emission_scores.device)
            for i, length in enumerate(lengths):
                assert emission_augmented[i, :length, :C_1].size() == emission_scores[i, :length].size()
                emission_augmented[i, :length, :C_1] = emission_scores[i, :length]
                emission_augmented[i, length, C_1] = 0
            # emission_augmented[:, :N_1, :C_1] = emission_scores
            # emission_augmented[:, lengths, C_1] = 0
            # emission_augmented[:, N_1, C_1] = 0

            lengths_augmented = lengths + 1

        else:
            transition_augmented = transition

            init_augmented = init

            length_scores_augmented = length_scores

            emission_augmented = emission_scores

            lengths_augmented = lengths

        scores = torch.zeros(b, N - 1, K, C, C, device=emission_scores.device).type_as(emission_scores)
        scores[:, :, :, :, :] += transition_augmented.view(b, 1, 1, C, C)
        # transition scores should include prior scores at first time step
        scores[:, 0, :, :, :] += init_augmented.view(b, 1, 1, C)
        scores[:, :, :, :, :] += length_scores_augmented.view(b, 1, K, 1, C)
        # add emission scores
        # TODO: progressive adding
        for k in range(1, K):
            # scores[:, :, k, :, :] += sliding_sum(emission_augmented, k).view(b, N, 1, C)[:, :N - 1]
            # scores[:, N - 1 - k, k, :, :] += emission_augmented[:, N - 1].view(b, C, 1)
            summed = sliding_sum(emission_augmented, k).view(b, N, 1, C)
            for i in range(b):
                length = lengths_augmented[i]
                scores[i, :length - 1, k, :, :] += summed[i, :length - 1]
                scores[i, length - 1 - k, k, :, :] += emission_augmented[i, length - 1].view(C, 1)

        return scores

    def add_eos(self, spans, lengths):
        b, N = spans.size()
        augmented = torch.cat([spans, torch.full([b, 1], -1, device=spans.device, dtype=torch.long)], dim=1)
        # assert (augmented[torch.arange(b), lengths] == -1).all()
        augmented[torch.arange(b), lengths] = self.n_classes
        return augmented

    def trim(self, spans, lengths, check_eos=False):
        # lengths should be the lengths NOT including any eos symbol at the end
        b, N = spans.size()
        indices = torch.arange(b)
        if check_eos:
            pass
            # if not (spans[indices, lengths] == self.n_classes).all():
            #     print("warning: EOS marker not present")
        seqs = []
        for i in range(b):
            seqs.append(spans[i, :lengths[i]])
        return seqs

    @property
    def batched_scores(self):
        return False

    def set_z(self, features, lengths, use_mean=False):
        # will be overridden by child
        self.kl = Variable(torch.zeros(features.size(0)).to(features.device), requires_grad=True)

    def score_features(self, features, lengths, valid_classes, add_eos, use_mean_z,
                       additional_allowed_ends_per_instance=None):
        # assert all_equal(lengths), "varied length scoring isn't implemented"
        # TODO: make this functional
        self.set_z(features, lengths, use_mean=use_mean_z)

        if self.feature_projector is not None:
            projected_features, jacobian = self.feature_projector(features)
        else:
            projected_features = features

        if self.allowed_ends is not None:
            # TODO: ugh
            if additional_allowed_ends_per_instance is None:
                additional_allowed_ends_per_instance = [set() for _ in features.size(0)]
            allowed_ends_per_instance = [
                [i for i, ix in enumerate(valid_classes)
                 if ix.item() in (set(self.allowed_ends) | set(additional_allowed_ends))]
                for additional_allowed_ends in additional_allowed_ends_per_instance
            ]
            assert all(allowed_ends_per_instance), allowed_ends_per_instance
        else:
            allowed_ends_per_instance = None

        return self.log_hsmm(
            self.transition_log_probs(valid_classes),
            self.emission_log_probs(projected_features, valid_classes),
            self.initial_log_probs(valid_classes),
            self.length_log_probs(valid_classes),
            lengths,
            add_eos=add_eos,
            all_batched=self.batched_scores,
            allowed_ends_per_instance=allowed_ends_per_instance,
        )

    def log_likelihood(self, features, lengths, valid_classes_per_instance, spans=None, add_eos=True, use_mean_z=False,
                       additional_allowed_ends_per_instance=None):
        if valid_classes_per_instance is not None:
            assert all_equal(set(vc.detach().cpu().numpy()) for vc in
                             valid_classes_per_instance), "must have same valid_classes for all instances in the batch"
            valid_classes = valid_classes_per_instance[0]
            C = len(valid_classes)
        else:
            valid_classes = None
            C = self.n_classes

        scores = self.score_features(features, lengths, valid_classes, add_eos=add_eos, use_mean_z=use_mean_z,
                                     additional_allowed_ends_per_instance=additional_allowed_ends_per_instance)

        K = scores.size(2)
        assert K <= self.max_k

        if add_eos:
            eos_lengths = lengths + 1
            eos_spans = self.add_eos(spans, lengths) if spans is not None else spans
            eos_C = C + 1
        else:
            eos_lengths = lengths
            eos_spans = spans
            eos_C = C

        dist = SemiMarkovCRF(scores, lengths=eos_lengths)

        if eos_spans is not None:
            eos_spans_mapped = eos_spans.detach().cpu().clone()
            if valid_classes is not None:
                # unmap
                mapping = {cls.item(): index for index, cls in enumerate(valid_classes)}
                assert len(mapping) == len(valid_classes), "valid_classes must be unique"
                assert -1 not in mapping
                mapping[-1] = -1
                mapping[self.n_classes] = C  # map EOS
                if 0 not in mapping:
                    # TODO: hack, 0 sometimes will signify padding
                    mapping[0] = 0
                eos_spans_mapped.apply_(lambda x: mapping[x])
            # features = features[:,:this_N,:]
            # spans = spans[:,:this_N]
            parts = SemiMarkovCRF.struct.to_parts(eos_spans_mapped, (eos_C, K),
                                                  lengths=eos_lengths).type_as(scores)

            # this maximizes p(x, y)
            d = parts.dim()
            batch_dims = range(d - len(dist.event_shape))
            log_likelihood = dist.struct().score(
                dist.log_potentials,
                parts.type_as(dist.log_potentials),
                batch_dims=batch_dims,
            ).mean()
            # this maximizes p(y | x)
            # log_likelihood = dist.log_prob(parts).mean()
        else:
            log_likelihood = dist.partition.mean()
        return log_likelihood

    def viterbi(self, features, lengths, valid_classes_per_instance, add_eos=True, use_mean_z=False,
                additional_allowed_ends_per_instance=None):
        if valid_classes_per_instance is not None:
            assert all_equal(set(vc.detach().cpu().numpy()) for vc in
                             valid_classes_per_instance), "must have same valid_classes for all instances in the batch"
            valid_classes = valid_classes_per_instance[0]
            C = len(valid_classes)
        else:
            valid_classes = None
            C = self.n_classes
        scores = self.score_features(features, lengths, valid_classes, add_eos=add_eos, use_mean_z=use_mean_z,
                                     additional_allowed_ends_per_instance=additional_allowed_ends_per_instance)
        if add_eos:
            eos_lengths = lengths + 1
        else:
            eos_lengths = lengths
        dist = SemiMarkovCRF(scores, lengths=eos_lengths)

        pred_spans, extra = dist.struct.from_parts(dist.argmax)
        # convert to class labels
        # pred_spans_trim = self.model.trim(pred_spans, lengths, check_eos=add_eos)

        pred_spans_unmap = pred_spans.detach().cpu()
        if valid_classes is not None:
            mapping = {index: cls.item() for index, cls in enumerate(valid_classes)}
            assert len(mapping.values()) == len(mapping), "valid_classes must be unique"
            assert -1 not in mapping.values()
            mapping[-1] = -1
            mapping[C] = self.n_classes  # map EOS
            # unmap
            pred_spans_unmap.apply_(lambda x: mapping[x])
        return pred_spans_unmap


class ComponentSemiMarkovModule(SemiMarkovModule):
    # portions of this code are adapted from Yoon Kim's Compound PCFG
    # github.com/harvardnlp/compound-pcfg/blob/master/models.py
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--sm_component_decompose_steps', action='store_true')
        parser.add_argument('--sm_component_mean_layers', type=int, default=2)
        parser.add_argument('--sm_component_length_layers', type=int, default=2)
        parser.add_argument('--sm_component_embedding_dim', type=int, default=100)
        # parser.add_argument('--sm_component_separate_embeddings', action='store_true')
        parser.add_argument('--sm_component_z_dim', type=int, default=0)
        parser.add_argument('--sm_component_z_hidden_dim', type=int, default=100)
        parser.add_argument('--no_sm_compound_structure', action='store_false', dest='sm_compound_structure')

    def __init__(self,
                 args,
                 n_classes: int,
                 n_components: int,
                 class_to_components: Dict[int, Set[int]],
                 feature_dim: int,
                 allow_self_transitions=False,
                 per_class_bias=True,
                 allowed_starts: Set[int] = None,
                 allowed_transitions: Dict[int, Set[int]] = None,
                 allowed_ends: Set[int] = None,
                 merge_classes: Dict[int, int] = None
                 ):
        assert merge_classes is None
        self.args = args
        self.n_components = n_components
        self.embedding_dim = self.args.sm_component_embedding_dim

        self.z_dim = self.args.sm_component_z_dim
        self.embedding_and_z_dim = self.embedding_dim + self.z_dim
        if self.args.sm_compound_structure:
            self.structure_emb_dim = self.embedding_and_z_dim
        else:
            self.structure_emb_dim = self.embedding_dim

        self.class_to_components = class_to_components

        self.component_to_classes: Dict[int, Set[int]] = {}
        for cls, components in self.class_to_components.items():
            for component in components:
                assert 0 <= component < n_components
                if component not in self.component_to_classes:
                    self.component_to_classes[component] = set()
                self.component_to_classes[component].add(cls)
        self.per_class_bias = per_class_bias
        self.mean_layers = self.args.sm_component_mean_layers
        self.length_layers = self.args.sm_component_length_layers
        # self.use_separate_embeddings = self.args.sm_component_separate_embeddings
        self.use_separate_embeddings = True
        super(ComponentSemiMarkovModule, self).__init__(args, n_classes, feature_dim, allow_self_transitions,
                                                        allowed_starts, allowed_transitions, allowed_ends)

    def init_params(self):
        # this initialization follows https://github.com/harvardnlp/compound-pcfg/blob/master/models.py
        # self.component_embeddings = nn.Parameter(torch.randn(self.n_components, self.embedding_dim))
        def make_embeddings():
            return nn.EmbeddingBag(num_embeddings=self.n_components,
                                   embedding_dim=self.embedding_dim,
                                   mode="mean",
                                   sparse=False)

        if self.use_separate_embeddings:
            self.initial_embeddings = make_embeddings()
            self.transition_embeddings = make_embeddings()
            self.emission_embeddings = make_embeddings()
            self.length_embeddings = make_embeddings()
        else:
            self.shared_embeddings = make_embeddings()

        # p(class) \propto exp(w \cdot embed(class) + b_class)
        self.initial_weights = nn.Linear(self.structure_emb_dim, 1, bias=True)
        if self.per_class_bias:
            self.initial_bias = nn.Parameter(torch.zeros(self.n_classes))
        else:
            self.initial_bias = None

        # p(class_2 | class_1) \propto exp(f(embed(class_1)) embed(class_2) + b_class_2)
        self.transition_weights = nn.Linear(self.structure_emb_dim, self.structure_emb_dim, bias=True)
        if self.per_class_bias:
            self.transition_bias = nn.Parameter(torch.zeros(self.n_classes))
        else:
            self.transition_bias = None

        emission_mean_mlp = [nn.Linear(self.embedding_and_z_dim, self.embedding_dim)]
        for _ in range(self.mean_layers):
            emission_mean_mlp.append(ResidualLayer(self.embedding_dim, self.embedding_dim))
        emission_mean_mlp.append(nn.Linear(self.embedding_dim, self.feature_dim))
        self.emission_mean_mlp = nn.Sequential(*emission_mean_mlp)
        self.emission_mean_bias = nn.Parameter(torch.zeros(self.feature_dim))

        length_mlp = [nn.Linear(self.structure_emb_dim, self.embedding_dim)]
        for _ in range(self.length_layers):
            length_mlp.append(ResidualLayer(self.embedding_dim, self.embedding_dim))
        length_mlp.append(nn.Linear(self.embedding_dim, 1))
        self.length_mlp = nn.Sequential(*length_mlp)

        if self.per_class_bias:
            self.length_bias = nn.Parameter(torch.zeros(self.n_classes))
        else:
            self.length_bias = None

        if self.z_dim != 0:
            self.encoder = Encoder(self.args, self.feature_dim, self.args.sm_component_z_hidden_dim)
            # times 2 for mean and log var
            self.encoder_to_params = nn.Linear(self.args.sm_component_z_hidden_dim, self.z_dim * 2)

        # shared, tied, diagonal covariance matrix
        gaussian_cov = torch.eye(self.feature_dim).float()
        self.gaussian_cov = nn.Parameter(gaussian_cov, requires_grad=False)

        for name, param in self.named_parameters():
            if param.dim() > 1 and name not in ['gaussian_cov']:
                xavier_uniform_(param)

    def flatten_parameters(self):
        if self.z_dim != 0:
            self.encoder.flatten_parameters()

    def _initialize_gaussian_means(self, mean):
        # self.gaussian_means.data = mean.expand((self.n_classes, self.n_dims))
        # TODO: better init that takes into account emission_mean_mlp
        self.emission_mean_bias.data.zero_()
        self.emission_mean_bias.data.add_(mean.squeeze(0))

    def fit_supervised(self, feature_list, label_list, state_smoothing=1e-2, length_smoothing=1e-1):
        raise NotImplementedError("closed form fit_supervised() not implemented for this model")

    def enc(self, features, lengths):
        # batch_size x max_length x z_hidden_dim
        encoded = self.encoder(features, lengths, output_padding_value=0)
        # batch_size x z
        params = self.encoder_to_params(encoded.max(1)[0])
        mean = params[:, :self.z_dim]
        logvar = params[:, self.z_dim:]
        return mean, logvar

    def _get_kl(self, mean, logvar):
        return -0.5 * (logvar - torch.pow(mean, 2) - torch.exp(logvar) + 1)

    def _get_z_and_kl(self, features, lengths, use_mean=False):
        batch_size = features.size(0)
        if self.z_dim > 0:
            mean, logvar = self.enc(features, lengths)
            z = mean.new(batch_size, self.z_dim).normal_(0, 1)
            z = (0.5 * logvar).exp() * z + mean
            kl = self._get_kl(mean, logvar).sum(1)
            if use_mean:
                z = mean
        else:
            z = torch.zeros(batch_size, 1, device=features.device)
            kl = torch.zeros(batch_size, device=features.device)
        return z, kl

    def set_z(self, features, lengths, use_mean=False):
        # z: batch_size x z_dim
        # kl: batch_size
        self.z, self.kl = self._get_z_and_kl(features, lengths, use_mean)

    def embed_classes(self, component_embeddings, valid_classes, is_structure):
        if valid_classes is None:
            valid_classes = torch.arange(self.n_classes, device=component_embeddings.weight.device)
        assert valid_classes.dim() == 1, valid_classes
        offset = 0
        offsets = [offset]
        indices = []
        for cls in valid_classes.detach().cpu().numpy():
            components = self.class_to_components[cls]
            offset += len(components)
            offsets.append(offset)
            indices.extend(components)
        assert offsets[-1] == len(indices), (offsets, indices)
        offsets = offsets[:-1]

        # len(valid_classes) x embedding_dim
        emb = component_embeddings(
            torch.tensor(indices, device=component_embeddings.weight.device, dtype=torch.long),
            torch.tensor(offsets, device=component_embeddings.weight.device, dtype=torch.long))
        n_valid_c, e_dim = emb.size()
        emb = emb.unsqueeze(0)
        if self.z_dim > 0 and (self.args.sm_compound_structure or not is_structure):
            assert hasattr(self, 'z'), 'make sure set_z has been called'
            batch, z_dim = self.z.size()
            emb = emb.expand(batch, n_valid_c, e_dim)
            z = self.z.unsqueeze(1).expand(batch, n_valid_c, z_dim)
            emb = torch.cat((emb, z), dim=-1)
        return emb

    @property
    def batched_scores(self):
        return True

    def initial_log_probs(self, valid_classes):
        # batch_size|1 x len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(
            self.initial_embeddings if self.use_separate_embeddings else self.shared_embeddings,
            valid_classes,
            is_structure=True
        )
        x = self.initial_weights(class_embeddings)
        x = x.squeeze(-1)
        if self.init_constraints is not None:
            constraints = self.init_constraints
            if valid_classes is not None:
                constraints = constraints[valid_classes]
            x = x.masked_fill(constraints.unsqueeze(0).expand_as(x), BIG_NEG)
        if self.initial_bias is not None:
            x += self.initial_bias[valid_classes]
        return torch.log_softmax(x, dim=-1)

    def transition_log_probs(self, valid_classes):
        # batch_size|1 x len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(
            self.transition_embeddings if self.use_separate_embeddings else self.shared_embeddings,
            valid_classes,
            is_structure=True
        )
        x = self.transition_weights(class_embeddings)
        x = torch.einsum("bfe,bte->btf", [x, class_embeddings])
        if self.transition_constraints is not None:
            constraints = self.transition_constraints
            if valid_classes is not None:
                constraints = constraints[valid_classes][:,valid_classes]
            x = x.masked_fill(constraints.unsqueeze(0).expand_as(x), BIG_NEG)
        if self.transition_bias is not None:
            x += self.transition_bias[valid_classes].unsqueeze(0).unsqueeze(-1).expand_as(x)
        if not self.allow_self_transitions:
            x = x.masked_fill(torch.eye(len(valid_classes), device=x.device).bool().unsqueeze(0).expand_as(x), BIG_NEG)
        assert x.dim() == 3
        # transition_logits are indexed: batch_size, to_state, from_state
        # so dim 1 should be normalized (in log-space)
        return F.log_softmax(x, dim=1)

    def emission_log_probs(self, features, valid_classes):
        # batch_size|1 x len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(
            self.emission_embeddings if self.use_separate_embeddings else self.shared_embeddings,
            valid_classes,
            is_structure=False
        )
        # batch_size|1 x len(valid_classes) x feature_dim
        class_means = self.emission_mean_mlp(class_embeddings)
        class_means += self.emission_mean_bias.unsqueeze(0).unsqueeze(0).expand_as(class_means)
        return self._emission_log_probs_with_means(features, class_means)

    def length_log_probs(self, valid_classes):
        # batch_size|1 x len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(
            self.length_embeddings if self.use_separate_embeddings else self.shared_embeddings,
            valid_classes,
            is_structure=True
        )
        # batch_size|1 x len(valid_classes)
        class_log_rates = self.length_mlp(class_embeddings).squeeze(-1)
        if self.length_bias is not None:
            class_log_rates += self.length_bias[valid_classes].unsqueeze(0).expand_as(class_log_rates)
        return self._length_log_probs_with_rates(class_log_rates)
