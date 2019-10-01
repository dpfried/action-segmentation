from torch_struct import SemiMarkov, MaxSemiring
import torch
import torch.nn.functional as F
import numpy as np
from semimarkov import SemimarkovModel
import itertools

#device = torch.device("cuda")
device = torch.device("cpu")

add_eos = False

sm_max = SemiMarkov(MaxSemiring)

b = 100
C = 7
N = 300
K = 50
step_length = 20

# b = 2
# C = 3
# N = 10
# K = 5
# step_length = 2

add_eos = True


num_steps = N // step_length
assert N % step_length == 0 # since we're fixing lengths, need to end perfectly

# trans_scores = torch.from_numpy(np.array([[0,1,0],[0,0,1],[1,0,0]]).T).float().log()
trans_scores = torch.zeros(C, C, device=device)
init_scores = torch.full((C,), -float("inf"), device=device)
init_scores[0] = 0

emission_scores = torch.full((b, N, C), -float("inf"), device=device)

for n in range(N):
    c = (n // step_length) % C
    emission_scores[:,n,c] = 1

length_scores = torch.full((K, C), -float("inf"), device=device)
length_scores[step_length,:] = 0

scores = SemimarkovModel.log_hsmm(trans_scores, emission_scores, init_scores, length_scores, add_eos=add_eos)
marginals = sm_max.marginals(scores)

sequence, extra = sm_max.from_parts(marginals)

for step in range(num_steps):
    c = step % C
    assert torch.allclose(sequence[:,step_length*step], torch.full((1,), c).long())

# C == EOS
if add_eos:
    assert torch.allclose(sequence[:,-1], torch.full((1,), C).long())
