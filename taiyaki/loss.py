import torch
from torch import nn

from taiyaki.layers import logaddexp


@torch.jit.script
def ctc_fwd_step(self, prev, xt, seqs):
    # Initialise with stay score
    score = xt[:, 4][:, None] + prev
    # Add move score
    move_score = torch.gather(xt, 1, seqs) + prev[:, :-1]
    score[:, 1:] = logaddexp(move_score, score[:, 1:])

    return score


class CTCLoss(nn.Module):
    def __init__(self, sharp=1.0):
        super().__init__()
        self.sharp = sharp

    def _fwd_step(self, prev, xt, seqs):
        # Initialise with stay score
        score = xt[:, 4][:, None] + prev
        # Add move score
        move_score = torch.gather(xt, 1, seqs) + prev[:, :-1]
        score[:, 1:] = logaddexp(move_score, score[:, 1:])

        return score

    def forward(self, x, seqs, seqlens):
        #  x is input tensor  T x B x 5
        #  seqs is list of sequence tensors
        nt, nb, nf = x.shape
        nbs, ns = seqs.shape
        assert nf == 5, "CTC requires 5 features, got {}".format(nf)
        assert nbs == nb, "Input and sequence batch size are inconsistent"
        assert len(seqlens) == nb, "Sequence length and batch size are inconsistent"

        #  Initialise forward vector. Must start in first position
        fwd = x.new_full((nb, ns + 1), -1e30)
        fwd[:, 0] = 0.0

        for xt in x.unbind(0):
            fwd = ctc_fwd_step(fwd, xt * self.sharp, seqs)
        return -torch.gather(fwd, 1, seqlens[:,None]) / (nt * self.sharp)


@torch.jit.script
def flipflop_step(prev, xt, move_idx, stay_idx):
    #  Initialise with stay score
    score = torch.gather(xt, 1, stay_idx) + prev
    #  Add on move score
    move_score = torch.gather(xt, 1, move_idx) + prev[:,:-1]
    score[:, 1:] = logaddexp(move_score, score[:,1:])

    return score


class FlipFlopLoss(nn.Module):
    def __init__(self, sharp=1.0):
        super().__init__()
        self.sharp = sharp

    def forward(self, x, move_idx, stay_idx, seqlens):
        #  x is input tensor  T x B x 5
        #  seqs is list of sequence tensors
        nt, nb, nf = x.shape
        nb_stay, npos = stay_idx.shape
        nb_move, npos_move = move_idx.shape

        assert nf == 40, "Flip-flop requires 40 features, got {}".format(nf)
        assert nb_stay == nb, "Input and stay index batch size are inconsistent"
        assert nb_move == nb, "Input and move index batch size are inconsistent"
        assert len(seqlens) == nb, "Sequence length and batch size are inconsistent"
        assert npos == npos_move + 1, "Move and stay indicies have different lengths"

        #  Initialise forward vector. Must start in first position
        fwd = x.new_full((nb, npos), -1e30)
        fwd[:, 0] = 0.0

        for xt in x.unbind(0):
            fwd = flipflop_step(fwd, xt * self.sharp, move_idx, stay_idx)
        return -torch.gather(fwd, 1, seqlens[:,None] - 1) / (self.sharp * nt)
