import numpy as np
import torch

from taiyaki.helpers import get_model_device, guess_model_stride


_DEFAULT_CHUNK_SIZE = 1000
_DEFAULT_OVERLAP = 100


def chunk_read(signal, chunk_size, overlap):
    """ Divide signal into overlapping chunks, trim if necessary

    Args:
        signal (:class:`ndarray`): Signal to split into chunks
        chunk_size (int): Length of chunks into which `signal` will be split.
        overlap (int): Overlap between one chunk and the next.

    Returns:
        tuple of :class:`ndarray` and :class:`ndarray` and :class:`ndarray`:
            Tensor containing chunked signal (chunk_size x nchunks x 1), an
            array containing the coordinate of the start position in the signal
            of each, and an array containing the coordinate of the end position
            in the signal (exclusive).

        Where the length of `signal` is less than `chunk_size`, then a single
        chunk of length equal to the length of `signal` is returned.
    """
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
    """ Stitch together neural network output or viterbi path from overlapping
    chunks

    Args:
        out (:class:`torch.Tensor`): Tensor containing output of network,
            dimensions time x batch x features.
        chunk_starts (:class:`ndarray`): array containing the coordinate of the
            start position of each chunk in the signal.
        chunk_ends (:class:`ndarray`): array containing the coordinate of the
            end position of each chunk in the signal (exclusive).
        stride (int): Stride of the model used to call `out`.
        path_stitching (bool): Include last value in stitching, default False.

    Returns:
        :class:`torch.Tensor`: A block x feature matrix containing the stitched
            chunks.
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
            end = (chunk_ends[i] + chunk_starts[i + 1] -
                   2 * chunk_starts[i]) // (2 * stride)
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

    Note:
        The `chunk_size` and `overlap` parameters as multiples of the stride of
    `model` rather than as number of samples.  This behaviour is consistent
    with the parameterisation in Guppy.

    Args:
        normed_signal (:class:`ndarray`): Signal of read, which will be chunked
            and the result of calling and stitching returned.
        model (:class:`layers.Serial`): A Taiyaki model, implicitly assumed to
            to have a :class:`layers.Serial` as its outmost layer and for the
            first wrapped layer to have parameters.
        chunk_size (int, optional): Length of chunks into which `signal` will
            be split.
        overlap (int, optional): Overlap between one chunk and the next.
        max_concur_chunks (int, optional): Calculate chunks in batches of size
            at most `max_concur_chunks`; if None, then all chunks are
            calculated at once.
        return_numpy (bool, optional): Return value should be converted
            :class:`ndarray` (default).
        return_tensor_on_device (bool, optional):  Return value should be moved
            back onto same device as model (default).  Overridden by
            `return_numpy`.

    Returns:
        :class:`Tensor`: Output of basecalling chunks, stitched together. If
            `return_tensor_on_device` is True, then the stitched chunks are
            transferred back on to the GPU device; otherwise, they are returned
            in host memory ("cpu" device).

        If `return_numpy` is True, the return type is converted to a
        :class:`ndarray` and remains in host memory.
    """
    device = get_model_device(model)
    stride = guess_model_stride(model)
    chunk_size *= stride
    overlap *= stride

    chunks, chunk_starts, chunk_ends = chunk_read(
        normed_signal, chunk_size, overlap)

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
