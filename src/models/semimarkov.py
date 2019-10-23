import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from sklearn.mixture import GaussianMixture
from torch import nn
from torch.distributions import MultivariateNormal, Poisson
from torch_struct import SemiMarkovCRF

from data.corpus import Datasplit
from models.model import Model, make_optimizer, make_data_loader
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
        self.n_dims = n_dims
        self.allow_self_transitions = allow_self_transitions
        poisson_log_rates = torch.zeros(n_classes).float()
        self.poisson_log_rates = nn.Parameter(poisson_log_rates, requires_grad=True)

        gaussian_means = torch.zeros(n_classes, n_dims).float()
        self.gaussian_means = nn.Parameter(gaussian_means, requires_grad=True)

        # shared, tied, diagonal covariance matrix
        gaussian_cov = torch.eye(n_dims).float()
        self.gaussian_cov = nn.Parameter(gaussian_cov, requires_grad=False)

        # target x source
        transition_logits = torch.zeros(n_classes, n_classes).float()
        self.transition_logits = nn.Parameter(transition_logits, requires_grad=True)

        init_logits = torch.zeros(n_classes).float()
        self.init_logits = nn.Parameter(init_logits, requires_grad=True)
        torch.nn.init.uniform_(self.init_logits, 0, 1)

        self.max_k = max_k

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
        self.init_logits.data.add_(torch.from_numpy(init_probs).log())

        smoothed_trans_counts = stats['span_transition_counts'] + state_smoothing

        trans_probs = smoothed_trans_counts / smoothed_trans_counts.sum(axis=0)[None, :]
        trans_probs[np.isnan(trans_probs)] = 0
        # to, from -- so rows should sum to 1
        # assert np.allclose(trans_probs.sum(axis=0), 1.0, rtol=1e-3), (trans_probs.sum(axis=0), trans_probs)
        self.transition_logits.data.zero_()
        self.transition_logits.data.add_(torch.from_numpy(trans_probs).log())

        mean_lengths = (stats['span_lengths'] + length_smoothing) / (stats['span_counts'] + length_smoothing)
        self.poisson_log_rates.data.zero_()
        self.poisson_log_rates.data.add_(torch.from_numpy(mean_lengths).log())

        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(torch.from_numpy(emission_gmm.means_).float())

        self.gaussian_cov.data.zero_()
        self.gaussian_cov.data.add_(torch.diag(torch.from_numpy(emission_gmm.covariances_[0]).float()))

    def initialize_gaussian_from_feature_list(self, features):
        feats = torch.cat(features, dim=0)
        assert feats.dim() == 2
        n_dim = feats.size(1)
        assert n_dim == self.n_dims
        mean = feats.mean(dim=0, keepdim=True)
        # self.gaussian_means.data = mean.expand((self.n_classes, self.n_dims))
        self.gaussian_means.data.zero_()
        self.gaussian_means.data.add_(mean.expand((self.n_classes, self.n_dims)))
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
        # so each row should be normalized (in log-space)
        return F.log_softmax(masked, dim=0)

    def emission_log_probs(self, features, valid_classes):
        b, N, d = features.size()
        feats_reshaped = features.reshape(-1, d)
        if valid_classes is None:
            class_indices = range(self.n_classes)
        else:
            class_indices = valid_classes
        dists = [
            MultivariateNormal(loc=self.gaussian_means[class_index],
                               covariance_matrix=self.gaussian_cov)
            for class_index in class_indices
        ]
        log_probs = [
            dist.log_prob(feats_reshaped).reshape(b, N, 1)  # b x
            for dist in dists
        ]
        return torch.cat(log_probs, dim=2)

    def length_log_probs(self, valid_classes):
        max_length = self.max_k
        if valid_classes is None:
            class_indices = list(range(self.n_classes))
            n_classes = self.n_classes
        else:
            class_indices = valid_classes
            n_classes = len(valid_classes)
        time_steps = torch.arange(max_length, device=self.poisson_log_rates.device).unsqueeze(-1).expand(max_length,
                                                                                                         n_classes).float()
        poissons = Poisson(torch.exp(self.poisson_log_rates[class_indices]))
        return poissons.log_prob(time_steps)

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
        #         for start in range(max(0, n - k + 1),n+1):
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
            assert (spans[indices, lengths] == self.n_classes).all()
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
            eos_spans_mapped = eos_spans.detach().cpu()
            if valid_classes is not None:
                # unmap
                mapping = {cls.item(): index for index, cls in enumerate(valid_classes)}
                assert len(mapping) == len(valid_classes), "valid_classes must be unique"
                assert -1 not in mapping
                mapping[-1] = -1
                mapping[self.n_classes] = C  # map EOS
                eos_spans_mapped.apply_(lambda x: mapping[x])
            # features = features[:,:this_N,:]
            # spans = spans[:,:this_N]
            parts = SemiMarkovCRF.struct.to_parts(eos_spans_mapped, (eos_C, K),
                                                  lengths=eos_lengths).type_as(scores)
            log_likelihood = dist.log_prob(parts).mean()
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


class SemiMarkovModel(Model):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--sm_max_span_length', type=int)
        parser.add_argument('--sm_supervised_state_smoothing', type=float, default=1e-2)
        parser.add_argument('--sm_supervised_length_smoothing', type=float, default=1e-1)
        parser.add_argument('--sm_supervised_gradient_descent', action='store_true')

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
        self.model = SemiMarkovModule(self.n_classes,
                                      self.feature_dim,
                                      max_k=self.args.sm_max_span_length,
                                      allow_self_transitions=True)
        if args.cuda:
            self.model.cuda()

    def fit_supervised(self, train_data: Datasplit):
        loader = make_data_loader(self.args, train_data, shuffle=False, batch_size=1)
        features, labels = [], []
        for batch in loader:
            features.append(batch['features'].squeeze(0))
            labels.append(batch['gt_single'].squeeze(0))
        self.model.fit_supervised(features, labels, self.args.sm_supervised_state_smoothing,
                                  self.args.sm_supervised_length_smoothing)

    def fit(self, train_data: Datasplit, use_labels: bool, callback_fn=None):
        self.model.train()
        if use_labels and not self.args.sm_supervised_gradient_descent:
            self.fit_supervised(train_data)
            callback_fn(0, {})
            return
        optimizer = make_optimizer(self.args, self.model.parameters())
        big_loader = make_data_loader(self.args, train_data, shuffle=True, batch_size=100)
        samp = next(iter(big_loader))
        big_features = samp['features']
        big_lengths = samp['lengths']
        if self.args.cuda:
            big_features = big_features.cuda()
            big_lengths = big_lengths.cuda()
        self.model.initialize_gaussian(big_features, big_lengths)

        loader = make_data_loader(self.args, train_data, shuffle=True, batch_size=self.args.batch_size)

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
        loader = make_data_loader(self.args, test_data, shuffle=False, batch_size=self.args.batch_size)
        for batch in loader:
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
                assert self.model.n_classes not in predictions[video], "predictions should not contain EOS: {}".format(predictions[video])
        return predictions
