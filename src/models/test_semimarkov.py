import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tqdm
from torch_struct import SemiMarkov, MaxSemiring
from torch_struct import SemiMarkovCRF

from models.semimarkov import SemiMarkovModule

# device = torch.device("cuda")
device = torch.device("cpu")

sm_max = SemiMarkov(MaxSemiring)

BIG_NEG = -1e9


class ToyDataset(Dataset):
    def __init__(self, labels, features, lengths):
        self.labels = labels
        self.features = features
        self.lengths = lengths

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, index):
        labels = self.labels[index]
        spans = SemiMarkovModule.labels_to_spans(labels.unsqueeze(0)).squeeze(0)
        return {
            'labels': self.labels[index],
            'features': self.features[index],
            'lengths': self.lengths[index],
            'spans': spans,
        }


def synthetic_data(num_data_points=200, C=3, N=100, K=5):
    def make_synthetic_features(class_labels, shift_constant=2.0):
        _batch_size, _N = class_labels.size()
        f = torch.randn((_batch_size, _N, C))
        shift = torch.zeros_like(f)
        shift.scatter_(2, class_labels.unsqueeze(2), shift_constant)
        return shift + f

    labels_l = []
    lengths = []
    for i in range(num_data_points):
        if i == 0:
            length = N
        else:
            length = random.randint(K, N)
        lengths.append(length)
        lab = []
        current_step = 0
        while len(lab) < N:
            step_length = random.randint(1, K-1)
            lab.extend([current_step % C] * step_length)
            current_step += 1
        lab = lab[:N]
        labels_l.append(lab)
    labels = torch.LongTensor(labels_l)
    features = make_synthetic_features(labels)
    lengths = torch.LongTensor(lengths)

    return labels, features, lengths


def partition_rows(arr, N):
    if isinstance(arr, list):
        assert N < len(list)
    else:
        assert N < arr.size(0)
    return arr[:N], arr[N:]


def test_learn_synthetic():
    C = 3
    K = 5
    N = 40
    N_train = 150
    N_test = 50

    epochs = 20

    train_data = ToyDataset(*synthetic_data(num_data_points=N_train, C=C, N=N, K=K))
    train_loader = DataLoader(train_data, batch_size=20)
    test_data = ToyDataset(*synthetic_data(num_data_points=N_test, C=C, N=N, K=K))
    test_loader = DataLoader(train_data, batch_size=20)

    model = SemiMarkovModule(C, C, max_k=K)
    model.initialize_gaussian(train_data.features, train_data.lengths)
    print(model.gaussian_means)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for epoch in range(epochs):
        losses = []
        for batch in train_loader:
            # if self.args.cuda:
            #     features = features.cuda()
            #     task_indices = task_indices.cuda()
            #     gt_single = gt_single.cuda()
            features = batch['features']
            lengths = batch['lengths']
            spans = batch['spans']
            this_N = lengths.max().item()
            features = features[:,:this_N,:]
            spans = spans[:,:this_N]
            scores = model.score_features(features, add_eos=True)
            dist = SemiMarkovCRF(scores, lengths=lengths+1)
            gold_parts = SemiMarkovCRF.struct.to_parts(model.add_eos(spans, lengths), (C+1, K), lengths=lengths+1).type_as(scores)
            this_loss = -dist.log_prob(gold_parts).mean()
            this_loss.backward()

            losses.append(this_loss.item())

            optimizer.step()
            model.zero_grad()
        train_acc, _ = predict_synthetic(model, train_loader)
        test_acc, _ = predict_synthetic(model, test_loader)
        print("epoch {} avg loss: {:.4f}\ttrain acc: {:.2f}\ttest acc: {:.2f}".format(epoch, np.mean(losses), train_acc, test_acc))

    return model, train_loader, test_loader


def predict_synthetic(model, dataloader):
    items = []
    token_match = 0
    token_total = 0
    for batch in dataloader:
        features = batch['features']
        lengths = batch['lengths']
        gold_spans = batch['spans']

        batch_size = features.size(0)

        this_N = lengths.max().item()
        features = features[:,:this_N,:]
        gold_spans = gold_spans[:,:this_N]

        scores = model.score_features(features, add_eos=True)
        dist = SemiMarkovCRF(scores, lengths=lengths+1)
        pred_spans, extra = dist.struct.from_parts(dist.argmax)

        gold_labels = SemiMarkovModule.spans_to_labels(gold_spans)
        pred_labels = SemiMarkovModule.spans_to_labels(pred_spans)

        gold_labels_trim = model.trim(gold_labels, lengths, check_eos=False)
        pred_labels_trim = model.trim(pred_labels, lengths, check_eos=True)

        assert len(gold_labels_trim) == batch_size
        assert len(pred_labels_trim) == batch_size

        for i in range(batch_size):
            item = {
                'length': lengths[i].item(),
                'gold_spans': gold_spans[i],
                'pred_spans': pred_spans[i],
                'gold_labels': gold_labels[i],
                'pred_labels': pred_labels[i],
                'gold_labels_trim': gold_labels_trim[i],
                'pred_labels_trim': pred_labels_trim[i],
            }
            items.append(item)
            token_match += (gold_labels_trim[i] == pred_labels_trim[i]).sum()
            token_total += pred_labels_trim[i].size(0)
    accuracy = 100.0 * token_match / token_total
    return accuracy, items


def test_labels_and_spans():
    position_labels = torch.LongTensor([[0, 1, 1, 2, 2, 2], [0, 1, 2, 3, 3, 4]])
    spans = torch.LongTensor([[0, 1, -1, 2, -1, -1], [0, 1, 2, 3, -1, 4]])
    assert (SemiMarkovModule.labels_to_spans(position_labels) == spans).all()
    assert (SemiMarkovModule.spans_to_labels(spans) == position_labels).all()

    rand_labels = torch.randint(low=0, high=3, size=(5,20))
    assert (SemiMarkovModule.spans_to_labels(SemiMarkovModule.labels_to_spans(rand_labels)) == rand_labels).all()


def test_log_hsmm():
    # b = 100
    # C = 7
    # N = 300
    # K = 50
    # step_length = 20

    b = 2
    C = 3
    N = 10
    K = 5
    step_length = 2

    add_eos = True

    padded_length = N + step_length * 2

    lengths_unpadded = torch.full((b,), N).long()
    lengths_unpadded[0] = padded_length
    lengths = lengths_unpadded + 1

    num_steps = N // step_length
    assert N % step_length == 0  # since we're fixing lengths, need to end perfectly

    # trans_scores = torch.from_numpy(np.array([[0,1,0],[0,0,1],[1,0,0]]).T).float().log()
    trans_scores = torch.zeros(C, C, device=device)
    init_scores = torch.full((C,), BIG_NEG, device=device)
    init_scores[0] = 0

    emission_scores = torch.full((b, padded_length, C), BIG_NEG, device=device)

    for n in range(padded_length):
        c = (n // step_length) % C
        emission_scores[:, n, c] = 1

    length_scores = torch.full((K, C), BIG_NEG, device=device)
    length_scores[step_length, :] = 0

    scores = SemiMarkovModule.log_hsmm(trans_scores, emission_scores, init_scores, length_scores, add_eos=add_eos)
    marginals = sm_max.marginals(scores, lengths=lengths)

    sequence, extra = sm_max.from_parts(marginals)

    for step in range(num_steps):
        c = step % C
        assert torch.allclose(sequence[:, step_length * step], torch.full((1,), c).long())

    # C == EOS
    if add_eos:
        batch_indices = torch.arange(0, b)
        assert torch.allclose(sequence[batch_indices, lengths - 1], torch.full((1,), C).long())


test_labels_and_spans()
print("test_labels_and_spans passed")

test_log_hsmm()
print("test_log_hsmm passed")

model, trainloader, testloader = test_learn_synthetic()
train_acc, train_preds = predict_synthetic(model, trainloader)
test_acc, test_preds = predict_synthetic(model, testloader)
print("train acc: {:.2f}".format(train_acc))
print("test acc: {:.2f}".format(test_acc))
