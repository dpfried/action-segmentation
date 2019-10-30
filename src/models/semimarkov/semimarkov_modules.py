import numpy as np
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.distributions import MultivariateNormal, Poisson
from torch_struct import SemiMarkovCRF

from typing import Dict, Set

from utils.utils import all_equal

BIG_NEG = -1e9


def get_diagonal_covariances(data):
    # data: num_points x feat_dim
    model = GaussianMixture(n_components=1, covariance_type='diag')
    responsibilities = np.ones((data.shape[0], 1))
    model._initialize(data, responsibilities)
    return model.covariances_, model.precisions_cholesky_


def semimarkov_sufficient_stats(feature_list, label_list, covariance_type, n_classes, max_k=None):
    assert len(feature_list) == len(label_list)
    tied_diag = covariance_type == 'tied_diag'
    if tied_diag:
        emissions = GaussianMixture(n_classes, covariance_type='diag')
    else:
        emissions = GaussianMixture(n_classes, covariance_type=covariance_type)
    X_l = []
    r_l = []

    span_counts = np.zeros(n_classes, dtype=np.float32)
    span_lengths = np.zeros(n_classes, dtype=np.float32)
    span_start_counts = np.zeros(n_classes, dtype=np.float32)
    # to, from
    span_transition_counts = np.zeros((n_classes, n_classes), dtype=np.float32)

    instance_count = 0

    # for i in tqdm.tqdm(list(range(len(train_data))), ncols=80):
    for X, labels in zip(feature_list, label_list):
        X_l.append(X)
        r = np.zeros((X.shape[0], n_classes))
        r[np.arange(X.shape[0]), labels] = 1
        assert r.sum() == X.shape[0]
        r_l.append(r)
        spans = SemiMarkovModule.labels_to_spans(labels.unsqueeze(0), max_k)
        # symbol, length
        rle_spans = SemiMarkovModule.rle_spans(spans, torch.LongTensor([spans.size(1)]))[0]
        last_symbol = None
        for index, (symbol, length) in enumerate(rle_spans):
            if index == 0:
                span_start_counts[symbol] += 1
            span_counts[symbol] += 1
            span_lengths[symbol] += length
            if last_symbol is not None:
                span_transition_counts[symbol, last_symbol] += 1
            last_symbol = symbol
        instance_count += 1

    X_arr = np.vstack(X_l)
    r_arr = np.vstack(r_l)
    emissions._initialize(X_arr, r_arr)
    if tied_diag:
        cov, prec_chol = get_diagonal_covariances(X_arr)
        emissions.covariances_[:] = np.copy(cov)
        emissions.precisions_cholesky_[:] = np.copy(prec_chol)
    return emissions, {
        'span_counts': span_counts,
        'span_lengths': span_lengths,
        'span_start_counts': span_start_counts,
        'span_transition_counts': span_transition_counts,
        'instance_count': instance_count,
    }


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


class SemiMarkovModule(nn.Module):
    def __init__(self, n_classes, n_dims, allow_self_transitions=False, max_k=None):
        super(SemiMarkovModule, self).__init__()
        self.n_classes = n_classes
        self.feature_dim = n_dims
        self.allow_self_transitions = allow_self_transitions
        self.init_params()
        self.max_k = max_k

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

    def fit_supervised(self, feature_list, label_list, state_smoothing=1e-2, length_smoothing=1e-1):
        emission_gmm, stats = semimarkov_sufficient_stats(
            feature_list, label_list,
            covariance_type='tied_diag',
            n_classes=self.n_classes,
            max_k=self.max_k,
        )
        init_probs = (stats['span_start_counts'] + state_smoothing) / float(
            stats['instance_count'] + state_smoothing * self.n_classes)
        init_probs[np.isnan(init_probs)] = 0
        # assert np.allclose(init_probs.sum(), 1.0), init_probs
        self.init_logits.data.zero_()
        self.init_logits.data.add_(torch.from_numpy(init_probs).to(device=self.init_logits.device).log())

        smoothed_trans_counts = stats['span_transition_counts'] + state_smoothing

        trans_probs = smoothed_trans_counts / smoothed_trans_counts.sum(axis=0)[None, :]
        trans_probs[np.isnan(trans_probs)] = 0
        # to, from -- so rows should sum to 1
        # assert np.allclose(trans_probs.sum(axis=0), 1.0, rtol=1e-3), (trans_probs.sum(axis=0), trans_probs)
        self.transition_logits.data.zero_()
        self.transition_logits.data.add_(torch.from_numpy(trans_probs).to(device=self.transition_logits.device).log())

        mean_lengths = (stats['span_lengths'] + length_smoothing) / (stats['span_counts'] + length_smoothing)
        self.poisson_log_rates.data.zero_()
        self.poisson_log_rates.data.add_(torch.from_numpy(mean_lengths).to(device=self.poisson_log_rates.device).log())

        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(
            torch.from_numpy(emission_gmm.means_).to(device=self.gaussian_means.device).float())

        self.gaussian_cov.data.zero_()
        self.gaussian_cov.data.add_(
            torch.diag(torch.from_numpy(emission_gmm.covariances_[0]).to(device=self.gaussian_cov.device).float()))

    def initialize_gaussian_from_feature_list(self, features):
        feats = torch.cat(features, dim=0)
        assert feats.dim() == 2
        n_dim = feats.size(1)
        assert n_dim == self.feature_dim
        mean = feats.mean(dim=0, keepdim=True)
        # self.gaussian_means.data = mean.expand((self.n_classes, self.n_dims))
        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(mean.expand((self.n_classes, self.feature_dim)))
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
        if valid_classes is not None:
            logits = logits[valid_classes]
        return F.log_softmax(logits, dim=0)

    def transition_log_probs(self, valid_classes):
        transition_logits = self.transition_logits
        if valid_classes is not None:
            transition_logits = transition_logits[valid_classes][:, valid_classes]
            n_classes = len(valid_classes)
        else:
            n_classes = self.n_classes
        if self.allow_self_transitions:
            masked = transition_logits
        else:
            masked = transition_logits.masked_fill(
                torch.eye(n_classes, device=self.transition_logits.device).bool(), BIG_NEG)
        # transition_logits are indexed: to_state, from_state
        # so each column should be normalized (in log-space)
        return F.log_softmax(masked, dim=0)

    def _emission_log_probs_with_means(self, features, class_means):
        num_classes, d_ = class_means.size()
        b, N, d = features.size()
        assert d == d_, (d, d_)
        feats_reshaped = features.reshape(-1, d)
        dists = [
            MultivariateNormal(loc=mean,
                               covariance_matrix=self.gaussian_cov)
            for mean in class_means
        ]
        log_probs = [
            dist.log_prob(feats_reshaped).reshape(b, N, 1)  # b x
            for dist in dists
        ]
        return torch.cat(log_probs, dim=2)

    def emission_log_probs(self, features, valid_classes):
        if valid_classes is None:
            class_indices = range(self.n_classes)
        else:
            class_indices = valid_classes
        class_means = self.gaussian_means[class_indices]
        return self._emission_log_probs_with_means(features, class_means)

    def _length_log_probs_with_rates(self, log_rates):
        n_classes, = log_rates.size()
        max_length = self.max_k
        time_steps = torch.arange(max_length, device=log_rates.device).unsqueeze(-1).expand(max_length, n_classes).float()
        poissons = Poisson(torch.exp(log_rates))
        return poissons.log_prob(time_steps)

    def length_log_probs(self, valid_classes):
        if valid_classes is None:
            class_indices = list(range(self.n_classes))
            n_classes = self.n_classes
        else:
            class_indices = valid_classes
            n_classes = len(valid_classes)
        log_rates = self.poisson_log_rates[class_indices]
        return self._length_log_probs_with_rates(log_rates)

    @staticmethod
    def labels_to_spans(position_labels, max_k):
        # position_labels: b x N, LongTensor
        assert not (position_labels == -1).any(), "position_labels already appear span encoded (have -1)"
        b, N = position_labels.size()
        last = position_labels[:, 0]
        values = [last.unsqueeze(1)]
        lengths = torch.ones_like(last)
        for n in range(1, N):
            this = position_labels[:, n]
            same_symbol = (last == this)
            if max_k is not None:
                same_symbol = same_symbol & (lengths < max_k - 1)
            encoded = torch.where(same_symbol, torch.full([1], -1, device=same_symbol.device, dtype=torch.long), this)
            lengths = torch.where(same_symbol, lengths, torch.full([1], 0, device=same_symbol.device, dtype=torch.long))
            lengths += 1
            values.append(encoded.unsqueeze(1))
            last = this
        return torch.cat(values, dim=1)

    @staticmethod
    def rle_spans(spans, lengths):
        b, T = spans.size()
        all_rle = []
        for i in range(b):
            this_rle = []
            this_spans = spans[i, :lengths[i]]
            current_symbol = None
            count = 0
            for symbol in this_spans:
                symbol = symbol.item()
                if current_symbol is None or symbol != -1:
                    if current_symbol is not None:
                        assert count > 0
                        this_rle.append((current_symbol, count))
                    count = 0
                    current_symbol = symbol
                count += 1
            if current_symbol is not None:
                assert count > 0
                this_rle.append((current_symbol, count))
            assert sum(count for sym, count in this_rle) == lengths[i]
            all_rle.append(this_rle)
        return all_rle

    @staticmethod
    def spans_to_labels(spans):
        # spans: b x N, LongTensor
        # contains 0.. for the start of a span (B-*), and -1 for its continuation (I-*)
        b, N = spans.size()
        current_labels = spans[:, 0]
        assert (current_labels != -1).all()
        values = [current_labels.unsqueeze(1)]
        for n in range(1, N):
            this = spans[:, n]
            this_labels = torch.where(this == -1, current_labels, this)
            values.append(this_labels.unsqueeze(1))
            current_labels = this_labels
        return torch.cat(values, dim=1)

    @staticmethod
    def log_hsmm(transition, emission_scores, init, length_scores, lengths, add_eos):
        """
        Convert HSMM to a linear chain.
        Parameters:
            transition: C X C
            emission_scores: b x N x C
            init: C
            length_scores: K x C
            add_eos: bool, whether to augment with an EOS class (with index C) which can only appear in the final timestep
        Returns:
            edges: b x (N-1) x C x C if not add_eos, or b x (N) x (C+1) x (C+1) if add_eos
        """
        batch, N_1, C_1 = emission_scores.shape
        K, _C = length_scores.shape
        if K > N_1:
            K = N_1
            length_scores = length_scores[:K]
        # assert N_1 >= K
        assert C_1 == _C
        # need to add EOS token
        if add_eos:
            N = N_1 + 1
            C = C_1 + 1
        else:
            N = N_1
            C = C_1
        b = emission_scores.size(0)
        if add_eos:
            transition_augmented = torch.full((C, C), BIG_NEG, device=transition.device)
            transition_augmented[:C_1, :C_1] = transition
            # can transition from anything to EOS
            transition_augmented[C_1, :] = 0

            init_augmented = torch.full((C,), BIG_NEG, device=init.device)
            init_augmented[:C_1] = init

            length_augmented = torch.full((K, C), BIG_NEG, device=length_scores.device)
            length_augmented[:, :C_1] = length_scores
            # EOS must be length 1, although I don't think this is checked in the dp
            length_augmented[1, C_1] = 0

            emission_augmented = torch.full((b, N, C), BIG_NEG, device=emission_scores.device)
            for i, length in enumerate(lengths):
                assert emission_augmented[i, :length, :C_1].size() == emission_scores[i, :length].size()
                emission_augmented[i, :length, :C_1] = emission_scores[i, :length]
                emission_augmented[i, length, C_1] = 0
            # emission_augmented[:, :N_1, :C_1] = emission_scores
            # emission_augmented[:, lengths, C_1] = 0
            # emission_augmented[:, N_1, C_1] = 0

        else:
            transition_augmented = transition

            init_augmented = init

            length_augmented = length_scores

            emission_augmented = emission_scores

        scores = torch.zeros(batch, N - 1, K, C, C, device=emission_scores.device).type_as(emission_scores)
        scores[:, :, :, :, :] += transition_augmented.view(1, 1, 1, C, C)
        # transition scores should include prior scores at first time step
        scores[:, 0, :, :, :] += init_augmented.view(1, 1, 1, C)
        scores[:, :, :, :, :] += length_augmented.view(1, 1, K, 1, C)
        # add emission scores
        # TODO: progressive adding
        for k in range(1, K):
            # scores[:, :, k, :, :] += sliding_sum(emission_augmented, k).view(b, N, 1, C)[:, :N - 1]
            # scores[:, N - 1 - k, k, :, :] += emission_augmented[:, N - 1].view(b, C, 1)
            summed = sliding_sum(emission_augmented, k).view(b, N, 1, C)
            for i in range(b):
                length = lengths[i]
                scores[i, :length - 1, k, :, :] += summed[i, :length - 1]
                scores[i, length - 1 - k, k, :, :] += emission_augmented[i, length - 1].view(C, 1)

        # for n in range(N):
        #     for k in range(K):
        #         for start in range(max(0,en e k + 1),n+1):
        #             end = min(start+k,n)
        #             scores[:,start:end,k,:,:] += emission_augmented[:,n,:].view(b,1,1,1,1,C)
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
            if not (spans[indices, lengths] == self.n_classes).all():
                print("warning: EOS marker not present")
        seqs = []
        for i in range(b):
            seqs.append(spans[i, :lengths[i]])
        return seqs

    def score_features(self, features, lengths, valid_classes, add_eos):
        # assert all_equal(lengths), "varied length scoring isn't implemented"
        return self.log_hsmm(
            self.transition_log_probs(valid_classes),
            self.emission_log_probs(features, valid_classes),
            self.initial_log_probs(valid_classes),
            self.length_log_probs(valid_classes),
            lengths,
            add_eos=add_eos,
        )

    def log_likelihood(self, features, lengths, valid_classes_per_instance, spans=None, add_eos=True):
        if valid_classes_per_instance is not None:
            assert all_equal(set(vc.detach().cpu().numpy()) for vc in
                             valid_classes_per_instance), "must have same valid_classes for all instances in the batch"
            valid_classes = valid_classes_per_instance[0]
            C = len(valid_classes)
        else:
            valid_classes = None
            C = self.n_classes

        scores = self.score_features(features, lengths, valid_classes, add_eos=add_eos)

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

    def viterbi(self, features, lengths, valid_classes_per_instance, add_eos=True):
        if valid_classes_per_instance is not None:
            assert all_equal(set(vc.detach().cpu().numpy()) for vc in
                             valid_classes_per_instance), "must have same valid_classes for all instances in the batch"
            valid_classes = valid_classes_per_instance[0]
            C = len(valid_classes)
        else:
            valid_classes = None
            C = self.n_classes
        scores = self.score_features(features, lengths, valid_classes, add_eos=add_eos)
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

class ResidualLayer(nn.Module):
    def __init__(self, in_dim = 100, out_dim = 100):
        super(ResidualLayer, self).__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        return F.relu(self.lin2(F.relu(self.lin1(x)))) + x

class ComponentSemiMarkovModule(SemiMarkovModule):
    def __init__(self,
                 n_classes: int,
                 n_components: int,
                 class_to_components: Dict[int, Set[int]],
                 feature_dim: int,
                 embedding_dim: int,
                 allow_self_transitions=False, max_k=None,
                 per_class_bias=False):
        self.n_components = n_components
        self.embedding_dim = embedding_dim

        self.class_to_components = class_to_components

        self.component_to_classes: Dict[int, Set[int]] = {}
        for cls, components in self.class_to_components.items():
            for component in components:
                assert 0 <= component < n_components
                if component not in self.component_to_classes:
                    self.component_to_classes[component] = set()
                self.component_to_classes[component].add(cls)
        self.per_class_bias = per_class_bias
        super(ComponentSemiMarkovModule, self).__init__(n_classes, feature_dim, allow_self_transitions, max_k=max_k)

    def init_params(self):
        # this initialization follows https://github.com/harvardnlp/compound-pcfg/blob/master/models.py
        # self.component_embeddings = nn.Parameter(torch.randn(self.n_components, self.embedding_dim))
        self.component_embeddings = nn.EmbeddingBag(num_embeddings=self.n_components,
                                                    embedding_dim=self.embedding_dim,
                                                    mode="mean",
                                                    sparse=True)

        # p(class) \propto exp(w \cdot embed(class) + b_class)
        self.initial_weights = nn.Linear(self.embedding_dim, 1, bias=True)
        if self.per_class_bias:
            self.initial_bias = nn.Parameter(torch.zeros(self.n_classes))
        else:
            self.initial_bias = None

        # p(class_2 | class_1) \propto exp(f(embed(class_1)) embed(class_2) + b_class_2)
        self.transition_weights = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)
        if self.per_class_bias:
            self.transition_bias = nn.Parameter(torch.zeros(self.n_classes))
        else:
            self.transition_bias = None

        self.emission_mean_mlp = nn.Sequential(
            ResidualLayer(self.embedding_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, self.feature_dim)
        )

        self.length_mlp = nn.Sequential(
            ResidualLayer(self.embedding_dim, self.embedding_dim),
            nn.Linear(self.embedding_dim, 1)
        )
        if self.per_class_bias:
            self.length_bias = nn.Parameter(torch.zeros(self.n_classes))
        else:
            self.length_bias = None

        # shared, tied, diagonal covariance matrix
        gaussian_cov = torch.eye(self.feature_dim).float()
        self.gaussian_cov = nn.Parameter(gaussian_cov, requires_grad=False)


    def fit_supervised(self, feature_list, label_list, state_smoothing=1e-2, length_smoothing=1e-1):
        raise NotImplementedError()

    def initialize_gaussian_from_feature_list(self, features):
        # TODO: implement this. component means average (?) across classes, with identity transformation matrices?
        raise NotImplementedError()

    def embed_classes(self, valid_classes):
        if valid_classes is None:
            valid_classes = torch.arange(self.n_classes, device=self.component_embeddings.weight.device)
        assert valid_classes.dim() == 1, valid_classes
        offset = 0
        offsets = [offset]
        indices = []
        for cls in valid_classes:
            components = self.class_to_components[cls]
            offsets.append(len(components))
            for cmp in components:
                indices.append(cmp)
        assert offsets[-1] == len(indices)
        offsets = offsets[:-1]

        # len(valid_classes) x embedding_dim
        return self.component_embeddings(torch.LongTensor(indices, device=valid_classes.device),
                                         torch.LongTensor(offsets, device=valid_classes.device))

    def initial_log_probs(self, valid_classes):
        # len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(valid_classes)
        # len(valid_clases) x 1
        x = self.initial_weights(class_embeddings)
        x = x.squeeze(1)
        if self.initial_bias is not None:
            x += self.initial_bias[valid_classes]
        return torch.log_softmax(x, dim=0)

    def transition_log_probs(self, valid_classes):
        # p(class_2 | class_1) \propto exp(f(embed(class_1)) embed(class_2) + b_class_2)

        # len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(valid_classes)
        # len(valid_classes) x embedding_dim
        x = self.transition_weights(class_embeddings)
        x = torch.einsum("fe,te->tf", [x, class_embeddings])
        if self.transition_bias is not None:
            x += self.transition_bias.unsqueeze(1).expand([len(valid_classes), len(valid_classes)])
        if not self.allow_self_transitions:
            x = x.masked_fill(
                torch.eye(self.n_classes, device=x.device).bool(), BIG_NEG)
        # transition_logits are indexed: to_state, from_state
        # so each column should be normalized (in log-space)
        return F.log_softmax(x, dim=0)

    def emission_log_probs(self, features, valid_classes):
        # len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(valid_classes)
        # len(valid_classes) x feature_dim
        class_means = self.emission_mean_mlp(class_embeddings)
        return self._emission_log_probs_with_means(features, class_means)

    def length_log_probs(self, valid_classes):
        # len(valid_classes) x embedding_dim
        class_embeddings = self.embed_classes(valid_classes)
        # len(valid_classes)
        class_log_rates = self.length_mlp(class_embeddings).squeeze(1)
        if self.length_bias is not None:
            class_log_rates += self.length_bias[valid_classes]
        return self._length_log_probs_with_rates(class_log_rates)
