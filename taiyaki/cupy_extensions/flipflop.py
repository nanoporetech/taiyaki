import cupy as cp
import numpy as np
import torch
from torch.autograd import Function


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

  float* fwd = all_fwd + 2 * nbase * blockIdx.x;
  int fwd_stride = 2 * nbase * N;

  float* fact = all_fact + blockIdx.x;
  int fact_stride = N;

  int to_base = threadIdx.x;

  // t = 0
  fwd[to_base] = -log(2.0 * nbase);
  fwd[to_base + nbase] = -log(2.0 * nbase);
  fwd += fwd_stride;
  fact[0] = log(2.0 * nbase);
  fact += N;
  __syncthreads();

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
    index = scores.device.index
    T, N, S = scores.shape
    nbase = int(np.sqrt(S / 2))

    fwd = torch.zeros((T + 1, N, 2 * nbase), dtype=scores.dtype, device=scores.device)
    fact = torch.zeros((T + 1, N, 1), dtype=scores.dtype, device=scores.device)
    with cp.cuda.Device(index):
        _flipflop_fwd(grid=(N, 1, 1), block=(nbase, 1, 1), args=(
            scores.data_ptr(), fwd.data_ptr(), fact.data_ptr(), T, N, nbase))
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
    index = scores.device.index
    T, N, S = scores.shape
    nbase = int(np.sqrt(S / 2))

    bwd = torch.zeros((T + 1, N, 2 * nbase), dtype=scores.dtype, device=scores.device)
    fact = torch.zeros((T + 1, N, 1), dtype=scores.dtype, device=scores.device)
    with cp.cuda.Device(index):
        _flipflop_bwd(grid=(N, 1, 1), block=(2 * nbase, 1, 1),
                      args=(scores.data_ptr(), bwd.data_ptr(), fact.data_ptr(), T, N, nbase))
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
    index = scores.device.index
    T, N, S = scores.shape
    nbase = int(np.sqrt(S / 2))
    fwd, fwd_fact = flipflop_fwd(scores)
    bwd, bwd_fact = flipflop_bwd(scores)
    trans = torch.zeros_like(scores)
    kernel_args = (
        scores.data_ptr(),
        fwd.data_ptr(),
        bwd.data_ptr(),
        trans.data_ptr(),
        T, N, nbase,
    )
    with cp.cuda.Device(index):
        _flipflop_make_trans(grid=(N,), block=(2 * nbase,), args=kernel_args)
    return trans, fwd_fact, bwd_fact


class LogZ(Function):

    @staticmethod
    def forward(ctx, scores):
        T, N, S = scores.shape
        trans, fwd_fact, bwd_fact = flipflop_make_trans(scores)
        ctx.save_for_backward(trans)
        return bwd_fact.sum(0)[:, 0]

    @staticmethod
    def backward(ctx, g):
        trans, = ctx.saved_tensors
        return trans.softmax(2) * g[:, None]


def logz(scores):
    return LogZ.apply(scores)


def global_norm(scores):
    return scores - logz(scores)[:, None] / len(scores)
