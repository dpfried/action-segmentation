import torch
from torch import nn
import torch.nn.functional as F

from torch.distributions import MultivariateNormal, Poisson

from models.model import Model, make_optimizer, make_data_loader

from data.corpus import Datasplit

BIG_NEG = -1e9

def sliding_sum(inputs, k):
    # inputs: b x T x c
    # sums sliding windows along the T dim, of length k
    batch_size = inputs.size(0)
    assert k > 0
    if k == 1:
        return inputs
    sliding_windows = F.unfold(inputs.unsqueeze(1),
                       kernel_size=(k,1),
                       padding=(k, 0)).reshape(batch_size, k, -1, inputs.size(-1))
    sliding_summed = sliding_windows.sum(dim=1)
    ret = sliding_summed[:,k:-1,:]
    assert ret.shape == inputs.shape
    return ret

class SemiMarkovModule(nn.Module):
    def __init__(self, n_classes, n_dims, max_k=None):
        super(SemiMarkovModule, self).__init__()
        self.n_classes = n_classes
        self.n_dims = n_dims
        poisson_log_rates = torch.zeros(n_classes).float()
        self.poisson_log_rates = nn.Parameter(poisson_log_rates, requires_grad=True)

        gaussian_means = torch.zeros(n_classes, n_dims).float()
        self.gaussian_means = nn.Parameter(gaussian_means, requires_grad=True)

        # shared, tied, diagonal covariance matrix
        gaussian_cov = torch.eye(n_dims).float()
        self.gaussian_cov = nn.Parameter(gaussian_cov, requires_grad=False)

        # source x target
        transition_logits = torch.zeros(n_classes, n_classes).float()
        self.transition_logits = nn.Parameter(transition_logits, requires_grad=True)

        init_logits = torch.zeros(n_classes).float()
        self.init_logits = nn.Parameter(init_logits, requires_grad=True)
        torch.nn.init.uniform_(self.init_logits, 0, 1)

        self.max_k = max_k

    def initialize_gaussian(self, data, lengths):
        # batch_size x n_dim
        batch_size, N, n_dim = data.size()
        assert lengths.size(0) == batch_size
        assert n_dim == self.n_dims
        feats = []
        for i in range(batch_size):
            feats.append(data[i,:lengths[i]])
        feats = torch.cat(feats, dim=0)
        assert feats.dim() == 2
        assert feats.size(1) == n_dim
        mean = feats.mean(dim=0, keepdim=True)
        self.gaussian_means.data = mean.expand((self.n_classes, self.n_dims))

        # TODO: consider using the biased estimator, with torch >= 1.2?
        self.gaussian_cov.data = torch.diag(feats.var(dim=0))

    def initial_log_probs(self):
        return F.log_softmax(self.init_logits)

    def transition_log_probs(self):
        masked = self.transition_logits.masked_fill(torch.eye(self.n_classes, device=self.transition_logits.device).byte(), BIG_NEG)
        # transition_logits are indexed: to_state, from_state
        # so each row should be normalized (in log-space)
        return F.log_softmax(masked, dim=0)

    def emission_log_probs(self, features):
        b, N, d = features.size()
        feats_reshaped = features.reshape(-1, d)
        dists = [
            MultivariateNormal(loc=mean, covariance_matrix=self.gaussian_cov)
            for mean in self.gaussian_means
        ]
        log_probs = [
            dist.log_prob(feats_reshaped).reshape(b, N, 1) # b x
            for dist in dists
        ]
        return torch.cat(log_probs, dim=2)

    def length_log_probs(self):
        max_length = self.max_k
        time_steps = torch.arange(max_length).unsqueeze(-1).expand(max_length, self.n_classes).float()
        poissons = Poisson(torch.exp(self.poisson_log_rates))
        return poissons.log_prob(time_steps)

    @staticmethod
    def labels_to_spans(position_labels):
        # position_labels: b x N, LongTensor
        # lengths: b, LongTensor
        assert not (position_labels == -1).any(), "position_labels already appear span encoded (have -1)"
        b, N = position_labels.size()
        last = position_labels[:,0]
        values = [last.unsqueeze(1)]
        for n in range(1, N):
            this = position_labels[:,n]
            encoded = torch.where(last == this, torch.LongTensor([-1]), this)
            values.append(encoded.unsqueeze(1))
            last = this
        return torch.cat(values, dim=1)

    @staticmethod
    def log_hsmm(transition, emission_scores, init, length_scores, add_eos):
        """
        Convert HSMM to a linear chain.
        Parameters:
            transition: C X C
            emission_scores: b x N x C
            init: C
            length_scores: K x C
        Returns:
            edges: b x (N-1) x C x C
        """
        batch, N_1, C_1 = emission_scores.shape
        K, _C = length_scores.shape
        assert N_1 >= K
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
            transition_augmented[C_1,:] = 0

            init_augmented = torch.full((C,), BIG_NEG, device=init.device)
            init_augmented[:C_1] = init

            length_augmented = torch.full((K, C), BIG_NEG, device=length_scores.device)
            length_augmented[:,:C_1] = length_scores
            # EOS must be length 1, although I don't think this is checked in the dp
            length_augmented[1,C_1] = 0

            emission_augmented = torch.full((b, N, C), BIG_NEG, device=emission_scores.device)
            emission_augmented[:,:N_1,:C_1] = emission_scores
            emission_augmented[:,N_1,C_1] = 0

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
            scores[:,:,k,:,:] += sliding_sum(emission_augmented, k).view(b,N,1,C)[:,:N-1]
            scores[:,N-1-k,k,:,:] += emission_augmented[:,N-1].view(b,C,1)

        # for n in range(N):
        #     for k in range(K):
        #         for start in range(max(0, n - k + 1),n+1):
        #             end = min(start+k,n)
        #             scores[:,start:end,k,:,:] += emission_augmented[:,n,:].view(b,1,1,1,1,C)
        return scores

    def add_eos(self, spans, lengths):
        b, N = spans.size()
        augmented = torch.cat([spans, torch.full((b,), -1).long().unsqueeze(-1)], dim=1)
        # assert (augmented[torch.arange(b), lengths] == -1).all()
        augmented[torch.arange(b),lengths] = self.n_classes
        return augmented

    def score_features(self, features, add_eos):
        return self.log_hsmm(
            self.transition_log_probs(),
            self.emission_log_probs(features),
            self.initial_log_probs(),
            self.length_log_probs(),
            add_eos=add_eos,
        )

class SemimarkovModel(Model):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--gm_covariance', choices=['full', 'diag', 'tied', 'tied_diag'], default='tied_diag')

    @classmethod
    def from_args(cls, args, train_data):
        n_classes = train_data._corpus.n_classes
        feature_dim = train_data.feature_dim
        return SemimarkovModel(args, n_classes, feature_dim)

    def __init__(self, args, n_classes, feature_dim):
        self.args = args
        self.n_classes = n_classes
        self.feature_dim = feature_dim
        self.model = None

    def fit(self, train_data: Datasplit, use_labels=False, callback_fn=None):
        pass
