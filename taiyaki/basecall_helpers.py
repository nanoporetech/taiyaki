import numpy as np
import torch

from taiyaki.helpers import guess_model_stride


_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_OVERLAP = 100


def chunk_read(signal, chunk_size, overlap):
    """ Divide signal into overlapping chunks """
    if len(signal) < chunk_size:
        return signal[:, None, None], np.array([0]), np.array([len(signal)])

    chunk_ends = np.arange(chunk_size, len(
        signal), chunk_size - overlap, dtype=int)
    chunk_ends = np.concatenate([chunk_ends, [len(signal)]], 0)
    chunk_starts = chunk_ends - chunk_size
    nchunks = len(chunk_ends)

    chunks = np.empty((chunk_size, nchunks, 1), dtype='f4')
    for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
        chunks[:, i, 0] = signal[start:end]

    # We will use chunk_starts and chunk_ends to stitch the basecalls together
    return chunks, chunk_starts, chunk_ends


def stitch_chunks(out, chunk_starts, chunk_ends, stride, path_stitching=False):
    """ Stitch together neural network output or viterbi path
    from overlapping chunks
    """
    nchunks = out.shape[1]

    if nchunks == 1:
        return out[:, 0]
    else:
        # first chunk
        start = chunk_starts[0] // stride
        end = (chunk_ends[0] + chunk_starts[1]) // (2 * stride)
        if path_stitching:
            end += 1
        stitched_out = [out[start:end, 0]]

        # middle chunks
        for i in range(1, nchunks - 1):
            start = (chunk_ends[i - 1] - chunk_starts[i]) // (2 * stride)
            end = (chunk_ends[i] + chunk_starts[i + 1]
                   - 2 * chunk_starts[i]) // (2 * stride)
            if path_stitching:
                start += 1
                end += 1
            stitched_out.append(out[start:end, i])

        # last chunk
        start = (chunk_ends[-2] - chunk_starts[-1]) // (2 * stride)
        end = (chunk_ends[-1] - chunk_starts[-1]) // stride
        if path_stitching:
            start += 1
            end += 1
        stitched_out.append(out[start:end, -1])

        return torch.cat(stitched_out, 0)


def run_model(
        normed_signal, model, chunk_size=_DEFAULT_CHUNK_SIZE,
        overlap=_DEFAULT_OVERLAP, max_concur_chunks=None, return_numpy=True,
        return_tensor_on_device=True):
    """ Hook for megalodon to run network via taiyaki
    """
    device = next(model.parameters()).device
    stride = guess_model_stride(model)
    chunk_size *= stride
    overlap *= stride

    chunks, chunk_starts, chunk_ends = chunk_read(
        normed_signal, chunk_size, overlap)
    device = next(model.parameters()).device
    chunks = torch.tensor(chunks)
    with torch.no_grad():
        if max_concur_chunks is None:
            out = model(chunks.to(device)).cpu()
        else:
            out = []
            for some_chunks in torch.split(chunks, max_concur_chunks, 1):
                out.append(model(some_chunks.to(device)).cpu())
            out = torch.cat(out, 1)
        stitched_chunks = stitch_chunks(
            out, chunk_starts, chunk_ends, stride)
    if return_numpy:
        return stitched_chunks.numpy()
    if return_tensor_on_device:
        return stitched_chunks.to(device)
    return stitched_chunks
