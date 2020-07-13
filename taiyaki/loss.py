import torch
from torch import nn

from taiyaki.layers import logaddexp


@torch.jit.script
def ctc_fwd_step(prev, xt, seqs):
    """ Forward step of CTC mapping calculation

    Args:
        prev (:class:`Tensor`): forwards scores for previous step, dimensions
            batch x position.
        xt (:class"`Tensor`): scores for each possible feature, dimensions
            batch x features
        seq (:class:`Tensor`): index of features occurring at each position of
            each sequence, dimensions batch x position.

    Returns:
        :class:`Tensor`: forward scores updated for current step.
    """
    # Initialise with stay score
    score = xt[:, 4][:, None] + prev
    # Add move score
    move_score = torch.gather(xt, 1, seqs) + prev[:, :-1]
    score[:, 1:] = logaddexp(move_score, score[:, 1:])

    return score


class CTCLoss(nn.Module):
    """ Calculate Neg-Log-Likelihood of sequence given CTC output

    Attributes:
        sharp (float): sharpening factor.
    """

    def __init__(self, sharp=1.0):
        """ Constructor for `CTCLoss`

        Args:
            sharp (float, optional): sharpening factor for mapping.
        """
        super().__init__()
        self.sharp = sharp

    def forward(self, x, seqs, seqlens):
        """ Forward method for CTC mapping calculation

        Args:
            x (:class:`Tensor`): Tensor with dimensions TBF containing scores
                for each possible observation (features) at every time point.
            seqs (:class"`Tensor`): Tensor with dimensions BP (positions)
                containing the index of the feature occurring at each position
                of each sequence.  Tensor is padded to the length of longest
                sequence.
            seqlens (:class:`Tensor`): 1D array containing the length of each
                sequence in `seqs`.

        Raises:
            AssertError: Dimensions are inconsistent

        Returns:
            :class:`Tensor`: A 1D array containing mapping score for each batch
                element.
        """
        #  x is input tensor  T x B x 5
        #  seqs is list of sequence tensors
        nt, nb, nf = x.shape
        nbs, ns = seqs.shape
        assert nf == 5, "CTC requires 5 features, got {}".format(nf)
        assert nbs == nb, "Input and sequence batch size are inconsistent"
        assert len(
            seqlens) == nb, "Sequence length and batch size are inconsistent"

        #  Initialise forward vector. Must start in first position
        fwd = x.new_full((nb, ns + 1), -1e30)
        fwd[:, 0] = 0.0

        for xt in x.unbind(0):
            fwd = ctc_fwd_step(fwd, xt * self.sharp, seqs)
        return -torch.gather(fwd, 1, seqlens[:, None]) / (nt * self.sharp)


@torch.jit.script
def flipflop_step(prev, xt, move_idx, stay_idx):
    """ Forward step of CTC mapping calculation

    Args:
        prev (:class:`Tensor`): forwards scores for previous step, dimensions
            batch x position.
        xt (:class"`Tensor`): scores for each possible transition, dimensions
            batch x ntransitions
        move_idx (:class:`Tensor`): index of transions for each position of
            each sequence that results in moving to the next position,
            dimensions batch x (position - 1).
        stay_idx (:class:`Tensor`): index of transions for each position of
            each sequence that results in staying in the current position,
            dimensions batch x position.

    Returns:
        :class:`Tensor`: forward scores updated for current step.
    """
    #  Initialise with stay score
    score = torch.gather(xt, 1, stay_idx) + prev
    #  Add on move score
    move_score = torch.gather(xt, 1, move_idx) + prev[:, :-1]
    score[:, 1:] = logaddexp(move_score, score[:, 1:])

    return score


class FlipFlopLoss(nn.Module):
    """ Calculate Neg-Log-Likelihood of sequence given Flip-Flop CRF output

    Attributes:
        sharp (float): sharpening factor.
    """

    def __init__(self, sharp=1.0):
        """ Constructor for `FlipFlopLoss`

        Args:
            sharp (float, optional): sharpening factor for mapping
        """
        super().__init__()
        self.sharp = sharp

    def forward(self, x, move_idx, stay_idx, seqlens):
        """ Forward method for flip-flop mapping calculation

        Args:
            x (:class:`Tensor`): Tensor with dimensions TBF containing scores
                for each possible observation (features) at every time point.
            move_idx (:class:`Tensor`): Tensor storing index of transions for
                each position of each sequence that results in moving to the
                next position, dimensions batch x (position - 1).
            stay_idx (:class:`Tensor`): Tensor storing index of transions for
                each position of each sequence that results in staying in the
                current position, dimensions batch x position.
            seqlens (:class:`Tensor`): 1D array containing the length of each
                sequence in `seqs`.

        Raises:
            AssertError: Dimensions are inconsistent

        Returns:
            :class:`Tensor`: A 1D array containing mapping score for each batch
                element.
        """
        #  x is input tensor  T x B x 5
        #  seqs is list of sequence tensors
        nt, nb, nf = x.shape
        nb_stay, npos = stay_idx.shape
        nb_move, npos_move = move_idx.shape

        assert nf == 40, "Flip-flop requires 40 features, got {}".format(nf)
        assert nb_stay == nb, "Input and stay index batch size are inconsistent"
        assert nb_move == nb, "Input and move index batch size are inconsistent"
        assert len(
            seqlens) == nb, "Sequence length and batch size are inconsistent"
        assert npos == npos_move + 1, "Move and stay indicies have different lengths"

        #  Initialise forward vector. Must start in first position
        fwd = x.new_full((nb, npos), -1e30)
        fwd[:, 0] = 0.0

        for xt in x.unbind(0):
            fwd = flipflop_step(fwd, xt * self.sharp, move_idx, stay_idx)
        return -torch.gather(fwd, 1, seqlens[:, None] - 1) / (self.sharp * nt)
