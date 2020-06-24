import cupy as cp
import numpy as np
import torch
from torch.autograd import Function


from taiyaki import flipflopfings
from taiyaki.constants import LARGE_VAL


_flipflop_fwd = cp.RawKernel(r'''
extern "C" __global__
void flipflop_fwd(
  const float* all_scores,
  float* all_fwd,
  float* all_fact,
  long long T,
  long long N,
  long long nbase
) {
  // all_scores is a (T, N, S) tensor where S = 2 * nbase * (nbase + 1)
  // all_fwd is the output tensor of shape (T + 1, N, 2 * nbase)
  // all_fwd should be filled with zeros
  // all_fact is a (T + 1, N) matrix of normalisation factors
  // all_scores and all_fwd should both be contiguous
  // a 1D grid parallelises over elements in the batch (N dimension)
  // a 1D threadpool parallelises over the fwd calculations for each base
  // should be launched with blockDim = (N, 1, 1) and threadDim = (nbase, 1, 1)

  int S = 2 * nbase * (nbase + 1);

  const float* scores = all_scores + S * blockIdx.x;
  int scores_stride = S * N;

  float* fwd = all_fwd + 2 * nbase * (N + blockIdx.x);
  int fwd_stride = 2 * nbase * N;

  float* fact = all_fact + N + blockIdx.x;
  int fact_stride = N;

  int to_base = threadIdx.x;

  float u, v;
  for (int t = 0; t < T; t++) {
    // to flip
    u = fwd[-fwd_stride] + scores[2 * nbase * to_base];
    for (int from_base = 1; from_base < 2 * nbase; from_base++) {
      v = fwd[from_base - fwd_stride] + scores[2 * nbase * to_base + from_base];
      u = max(u, v) + log1p(exp(-abs(u - v)));
    }
    fwd[to_base] = u;

    // to flop
    u = fwd[to_base - fwd_stride] + scores[2 * nbase * nbase + to_base];
    v = fwd[to_base + nbase - fwd_stride] + scores[(2 * nbase + 1) * nbase + to_base];
    fwd[to_base + nbase] = max(u, v) + log1p(exp(-abs(u - v)));
    __syncthreads();

    // calculate normalisation factor
    if (to_base == 0) {
      u = fwd[0];
      for (int from_base = 1; from_base < 2 * nbase; from_base++) {
        v = fwd[from_base];
        u = max(u, v) + log1p(exp(-abs(u - v)));
      }
      fact[0] = u;
    }
    __syncthreads();

    // normalise
    fwd[to_base] -= fact[0];
    fwd[to_base + nbase] -= fact[0];
    __syncthreads();

    //if (to_base == 0) {
        //fact[0] += fact[-N];
    //}

    scores += scores_stride;
    fwd += fwd_stride;
    fact += fact_stride;
  }
}
''', 'flipflop_fwd')


def flipflop_fwd(scores):
    """ Forward calculation for flipflop transitions from raw network output

    The forward matrix entries are:

        fwd[t, s] = logsumexp(scores of paths ending in state s at time t)

    Paths must start in a flip state at time 0.

    For numerical reasons, we calculate a row-normalised forward matrix
    and a vector of scaling factors:

        fwd'[t, s] = fwd[t, s] - logsumexp(fwd[t, s])
        fact[t] = logsumexp(fwd[t, s]) - logsumexp(fwd[t - 1, s])

    :param scores: a [T, B, S] tensor containing a batch of B scores matrices
        each with T blocks and S flipflop transitions scores, where
        S = 2 * nbase * (nbase + 1)

    :returns: (fwd, fact) tensors of shape [T + 1, N, 2 * nbase] and [T + 1, B, 1]
    """
    index = scores.device.index
    T, N, S = scores.shape
    nbase = flipflopfings.nbase_flipflop(S)

    fwd = torch.zeros(
        (T + 1, N, 2 * nbase), dtype=scores.dtype, device=scores.device)
    fwd[0, :, :nbase] = 0.0
    fwd[0, :, nbase:] = -LARGE_VAL
    fact = torch.zeros((T + 1, N, 1), dtype=scores.dtype, device=scores.device)
    with cp.cuda.Device(index):
        _flipflop_fwd(grid=(N, 1, 1), block=(nbase, 1, 1),
                      args=(scores.contiguous().data_ptr(), fwd.data_ptr(),
                            fact.data_ptr(), T, N, nbase))

    return fwd, fact


_flipflop_bwd = cp.RawKernel(r'''
extern "C" __global__
void flipflop_bwd(
  const float* all_scores,
  float* all_bwd,
  float* all_fact,
  long long T,
  long long N,
  long long nbase
) {
  // all_scores is a (T, N, S) tensor where S = 2 * nbase * (nbase + 1)
  // all_bwd is the output tensor of shape (T + 1, N, 2 * nbase)
  // all_bwd should be filled with zeros
  // all_fact is a (T + 1, N) matrix of normalisation factors
  // all_scores and all_bwd should both be contiguous
  // a 1D grid parallelises over elements in the batch (N dimension)
  // a 1D threadpool parallelises over the bwd calculations for each state
  // should be launched with blockDim = (N, 1, 1) and threadDim = (2 * nbase, 1, 1)

  int S = 2 * nbase * (nbase + 1);

  const float* scores = all_scores + S * ((T - 1) * N + blockIdx.x);
  int scores_stride = S * N;

  float* bwd = all_bwd + 2 * nbase * (T * N + blockIdx.x);
  int bwd_stride = 2 * nbase * N;

  float* fact = all_fact + T * N + blockIdx.x;
  int fact_stride = N;

  int from_base = threadIdx.x;
  int to_base;

  // t = T
  bwd[from_base] = -log(2.0 * nbase);
  bwd -= bwd_stride;
  fact[0] = log(2.0 * nbase);
  fact -= N;
  __syncthreads();

  float u, v;
  for (int t = 0; t < T; t++) {
    // to flip
    u = bwd[bwd_stride] + scores[from_base];
    for (int to_base = 1; to_base < nbase; to_base++) {
      v = bwd[to_base + bwd_stride] + scores[2 * nbase * to_base + from_base];
      u = max(u, v) + log1p(exp(-abs(u - v)));
    }
    // to flop
    to_base = (from_base < nbase) ? from_base + nbase : from_base;
    v = bwd[to_base + bwd_stride] + scores[2 * nbase * nbase + from_base];
    u = max(u, v) + log1p(exp(-abs(u - v)));

    bwd[from_base] = u;

    // calculate normalisation factor
    if (from_base == 0) {
      u = bwd[0];
      for (int to_base = 1; to_base < 2 * nbase; to_base++) {
        v = bwd[to_base];
        u = max(u, v) + log1p(exp(-abs(u - v)));
      }
      fact[0] = u;
    }
    __syncthreads();

    // normalise
    bwd[from_base] -= fact[0];
    __syncthreads();

    //if (from_base == 0) {
        //fact[0] += fact[N];
    //}

    scores -= scores_stride;
    bwd -= bwd_stride;
    fact -= fact_stride;
  }
}
''', 'flipflop_bwd')


def flipflop_bwd(scores):
    """ Backward calculation for flipflop transitions from raw network output

    The backward matrix entries are:

        bwd[t, s] = logsumexp(scores of paths starting in state s at time t)

    Paths can end in any state at time T.

    For numerical reasons, we calculate a row-normalised backward matrix
    and a vector of scaling factors:

        bwd'[t, s] = bwd[t, s] - logsumexp(bwd[t, s])
        fact[t] = logsumexp(bwd[t, s]) - logsumexp(bwd[t + 1, s])

    :param scores: a [T, B, S] tensor containing a batch of B scores matrices
        each with T blocks and S flipflop transitions scores, where
        S = 2 * nbase * (nbase + 1)

    :returns: (bwd, fact) tensors of shape [T + 1, N, 2 * nbase] and [T + 1, B, 1]
    """
    index = scores.device.index
    T, N, S = scores.shape
    nbase = flipflopfings.nbase_flipflop(S)

    bwd = torch.zeros(
        (T + 1, N, 2 * nbase), dtype=scores.dtype, device=scores.device)
    fact = torch.zeros((T + 1, N, 1), dtype=scores.dtype, device=scores.device)
    with cp.cuda.Device(index):
        _flipflop_bwd(grid=(N, 1, 1), block=(2 * nbase, 1, 1),
                      args=(scores.contiguous().data_ptr(), bwd.data_ptr(),
                            fact.data_ptr(), T, N, nbase))
    return bwd, fact


_flipflop_make_trans = cp.RawKernel(r'''
extern "C" __global__
void flipflop_make_trans(
  const float* scores,
  const float* fwd,
  const float* bwd,
  float* trans,
  long long T,
  long long N,
  long long nbase
) {
  // scores is a (T, N, S) tensor where S = 2 * nbase * (nbase + 1)
  // fwd is (T + 1, N, 2 * nbase) matrix of forward scores
  // bwd is (T + 1, N, 2 * nbase) matrix of backward scores
  // trans is of the same shape as scores
  // trans should be filled with zeros
  // all tensors should be contiguous
  // a 1D grid parallelises over elements in the batch (N dimension)
  // a 1D threadpool parallelises over the calculations for each base
  // should be launched with blockDim = (N, 1, 1) and threadDim = (2 * nbase, 1, 1)
  int S = 2 * nbase * (nbase + 1);

  int scores_offset = S * blockIdx.x;
  int scores_stride = S * N;
  int fwd_offset = 2 * nbase * blockIdx.x;
  int fwd_stride = 2 * nbase * N;

  int from_base = threadIdx.x;
  int to_base;

  float f, s, b;
  for (int t = 0; t < T; t++) {
    f = fwd[fwd_offset + from_base];
    for (int to_base = 0; to_base < nbase; to_base++) {
        b = bwd[fwd_offset + fwd_stride + to_base];
        s = scores[scores_offset + from_base + 2 * nbase * to_base];
        trans[scores_offset + from_base + 2 * nbase * to_base] = f + s + b;
    }
    to_base = (from_base < nbase) ? from_base + nbase : from_base;
    b = bwd[fwd_offset + fwd_stride + to_base];
    s = scores[scores_offset + 2 * nbase * nbase + from_base];
    trans[scores_offset + 2 * nbase * nbase + from_base] = f + s + b;
    scores_offset += scores_stride;
    fwd_offset += fwd_stride;
    __syncthreads();
  }
}
''', 'flipflop_make_trans')


def flipflop_make_trans(scores):
    """ Calculates posterior transition probabilities from flipflop scores

    The posterior transition probabilities matrix is defined as:

        trans[t, uv] = probability of paths that use transition uv at time t

    Paths must start in a flip state at time 0.

    :param scores: a [T, B, S] tensor containing a batch of B scores matrices
        each with T blocks and S flipflop transitions scores, where
        S = 2 * nbase * (nbase + 1)

    :returns: (trans, fwd_fact, bwd_fact) tensors of shape [T, N, S],
        [T + 1, B] and [T + 1, B, 1]. fwd_fact and bwd_fact are the scaling
        factors returned by flipflop_fwd and flopflop_bwd
    """
    index = scores.device.index
    T, N, S = scores.shape
    nbase = flipflopfings.nbase_flipflop(S)

    fwd, fwd_fact = flipflop_fwd(scores)
    bwd, bwd_fact = flipflop_bwd(scores)
    scores = scores.contiguous()
    trans = torch.zeros_like(scores)
    kernel_args = (
        scores.contiguous().data_ptr(),
        fwd.data_ptr(),
        bwd.data_ptr(),
        trans.data_ptr(),
        T, N, nbase,
    )
    with cp.cuda.Device(index):
        _flipflop_make_trans(grid=(N,), block=(2 * nbase,), args=kernel_args)
    return trans, fwd_fact, bwd_fact


class LogZ(Function):
    """ Calculate the log partition function for a flipflop CRF """

    @staticmethod
    def forward(ctx, scores):
        T, N, S = scores.shape
        trans, fwd_fact, bwd_fact = flipflop_make_trans(scores)
        ctx.save_for_backward(trans)
        return fwd_fact.sum(0)[:, 0]

    @staticmethod
    def backward(ctx, g):
        trans, = ctx.saved_tensors
        return trans.softmax(2) * g[:, None]


def logz(scores):
    """ Calculate the log partition function for a flipflop CRF

    :param scores: a [T, B, S] tensor containing a batch of B scores matrices
        each with T blocks and S flipflop transitions scores, where
        S = 2 * nbase * (nbase + 1)

    :returns: logZ vector of shape [N]
    """
    return LogZ.apply(scores)


def global_norm(scores):
    """ Globally normalise flipflop CRF scores

    The globally normed scores satisfy: logz(normed scores) = 1

    :param scores: a [T, B, S] tensor containing a batch of B scores matrices
        each with T blocks and S flipflop transitions scores, where
        S = 2 * nbase * (nbase + 1)

    :returns: normed_scores tensor of shape [T, N, S]
    """
    return scores - logz(scores)[:, None] / len(scores)


_flipflop_viterbi = cp.RawKernel(r'''
extern "C" __global__
void flipflop_viterbi(
  const float* scores,
  float* fwd,
  unsigned long long* tb,
  unsigned long long* bp,
  long long T,
  long long N,
  long long nbase
) {
  // scores is a (T,  N, S) tensor where S = 2 * nbase * (nbase + 1)
  // fwd is the output tensor of shape (T + 1, N, 2 * nbase)
  // fwd should be filled with zeros
  // tb is the output tensor of shape (T + 1, N, 2 * nbase)
  // tb should be filled with zeros
  // bp is a (T + 1, N) tensor in which the best paths are written
  // scores and fwd should both be contiguous
  // a 1D grid parallelises over elements in the batch (N dimension)
  // a 1D threadpool parallelises over the fwd calculations for each base
  // should be launched with blockDim = (N, 1, 1) and threadDim = (nbase, 1, 1)
  int S = 2 * nbase * (nbase + 1);

  int in_batch_offset = S * blockIdx.x;
  int flip_offset = in_batch_offset + 2 * nbase * threadIdx.x;
  int flop_offset = in_batch_offset + 2 * nbase * nbase + threadIdx.x;
  int in_stride = S * N;

  int out_batch_offset = 2 * nbase * (N + blockIdx.x);
  int out_prev_offset = 2 * nbase * blockIdx.x;
  int out_offset = out_batch_offset + threadIdx.x;
  int out_stride = 2 * nbase * N;

  float u, v;
  int s;
  for (int t = 0; t < T; t++) {
    s = 0;
    u = scores[flip_offset] + fwd[out_prev_offset];
    for (int i = 1; i < 2 * nbase; i++) {
      v = scores[flip_offset + i] + fwd[out_prev_offset + i];
      if (v > u) {
        u = v;
        s = i;
      }
    }
    fwd[out_offset] = u;
    tb[out_offset] = s;
    u = scores[flop_offset] + fwd[out_prev_offset + threadIdx.x];
    v = scores[flop_offset + nbase] + fwd[out_prev_offset + threadIdx.x + nbase];
    fwd[out_offset + nbase] = max(u, v);
    tb[out_offset + nbase] = (u > v) ? threadIdx.x : threadIdx.x + nbase;
    flip_offset += in_stride;
    flop_offset += in_stride;
    out_prev_offset += out_stride;
    out_offset += out_stride;
    __syncthreads();
  }

  // traceback
  int tb_offset = (T * N + blockIdx.x) * 2 * nbase;
  int bp_offset = T * N + blockIdx.x;
  if (threadIdx.x == 0) {
    u = fwd[tb_offset];
    s = 0;
    for (int i = 1; i < 2 * nbase; i++) {
      if (fwd[tb_offset + i] > u) {
        u = fwd[tb_offset + i];
        s = i;
      }
    }
    for (int t = T - 1; t >= 0; t--) {
      bp[bp_offset] = s;
      s = tb[tb_offset + s];
      tb_offset -= 2 * nbase * N;
      bp_offset -= N;
    }
    bp[bp_offset] = s;
  }
}
''', 'flipflop_viterbi')


def flipflop_viterbi(scores):
    """ Calculate the Viterbi path through flipflop scores matrix

    The scores are assumed to be in log space, i.e. the probability
    of a path is proportional to exp(sum of transition scores on path)

    The function returns the (Viterbi) forward matrix defined as:

        viterbi_fwd[t, s] = score of best path to time t and state s

    and a traceback matrix:

        traceback[t, s] = previous state on best path to time t and state s

    and a vector encoding the sequence of states on the best path.

    :param scores: a [T, B, S] tensor containing a batch of B scores matrices
        each with T blocks and S flipflop transitions scores, where
        S = 2 * nbase * (nbase + 1)

    :returns: (fwd, traceback, best_path) tensors of shapes [T + 1, N, 2 * nbase],
        [T + 1, N, 2 * nbase] and [T + 1, N]
    """
    index = scores.device.index
    T, N, S = scores.shape
    nbase = flipflopfings.nbase_flipflop(S)

    scores = scores.contiguous()
    fwd = torch.zeros((T + 1, N, 2 * nbase),
                      dtype=scores.dtype, device=scores.device)
    fwd[:1, :, nbase:] = -LARGE_VAL
    traceback = torch.zeros((T + 1, N, 2 * nbase),
                            dtype=torch.long, device=scores.device)
    best_path = torch.zeros((T + 1, N), dtype=torch.long, device=scores.device)
    with cp.cuda.Device(index):
        _flipflop_viterbi(
            grid=(N, 1, 1),
            block=(nbase, 1, 1),
            args=(
                scores.data_ptr(),
                fwd.data_ptr(),
                traceback.data_ptr(),
                best_path.data_ptr(),
                T, N, nbase
            )
        )
    return fwd, traceback, best_path
