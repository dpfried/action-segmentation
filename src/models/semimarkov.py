import torch
from torch import nn
import torch.nn.functional as F

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

class SemimarkovModel(nn.Module):
    def __init__(self, n_classes, n_dims, max_k=None):
        poisson_rates = torch.zeros(n_classes).float()
        self.poisson_rates = nn.Parameter(poisson_rates, requires_grad=True)

        gaussian_means = torch.zeros(n_classes, n_dims).float()
        self.gaussian_means = nn.Parameter(gaussian_means, requires_grad=True)

        # shared, tied, diagonal covariance matrix
        gaussian_cov = torch.eye(n_dims).float()
        self.gaussian_cov = nn.Parameter(gaussian_cov, requires_grad=False)

        # source x target
        transition_logits = torch.zeros(n_classes, n_classes).float()
        self.transition_logits = nn.Parameter(transition_logits, requires_grad=True)

        self.max_k = max_k

    def init_gaussian(self, data):
        # batch_size x n_dim
        assert data.size(1) == self.n_dims
        mean = torch.mean(data, dim=0)
        self.gaussian_means.fill_(mean.unsqueeze(0).expand((self.n_classes, self.n_dims)))

    def initial_log_probs(self):
        return F.log_softmax(self.init_state_logits)

    def transition_log_probs(self):
        masked = self.transition_logits.masked_fill(torch.eye(self.n_classes, device=self.transition_logits.device), -float("inf"))
        return F.log_softmax(masked, dim=1)

    def transition_probabilities(self):
        pass

    @staticmethod
    def log_hsmm(transition, emission_scores, init, length_scores, add_eos=False):
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
            transition_augmented = torch.full((C, C), -float("inf"), device=transition.device)
            transition_augmented[:C_1, :C_1] = transition
            # can transition from anything to EOS
            transition_augmented[C_1,:] = 0

            init_augmented = torch.full((C,), -float("inf"), device=init.device)
            init_augmented[:C_1] = init

            length_augmented = torch.full((K, C), -float("inf"), device=length_scores.device)
            length_augmented[:,:C_1] = length_scores
            # EOS must be length 1, although I don't think this is checked in the dp
            length_augmented[1,C_1] = 0

            emission_augmented = torch.full((b, N, C), -float("inf"), device=emission_scores.device)
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

    def score(self, observations, lengths):
        batch_size, max_length = observations.size()

        # b x N x K x (C+1) x C
        # batch x time x length x to x from
        scores = torch.zeros(batch_size, max_length+1, self.max_k, self.n_classes+1, self.n_classes)

        # initial state distributions
