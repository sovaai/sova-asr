from scipy import signal
import numpy as np
import scipy
from numpy import fft
from numpy.lib.stride_tricks import as_strided


MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10


def get_window(window, nx, fftbins=True):
    if callable(window):
        return window(nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        return scipy.signal.get_window(window, nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == nx:
            return np.asarray(window)

        raise Exception('Window size mismatch: '
                        '{:d} != {:d}'.format(len(window), nx))
    else:
        raise Exception('Invalid window specification: {}'.format(window))


def pad_center(data, size, axis=-1, **kwargs):
    kwargs.setdefault('mode', 'constant')

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise Exception(('Target size ({:d}) must be '
                         'at least input size ({:d})').format(size, n))

    return np.pad(data, lengths, **kwargs)


def frame(x, frame_length, hop_length, axis=-1):
    if not isinstance(x, np.ndarray):
        raise Exception('Input must be of type numpy.ndarray, '
                        'given type(x)={}'.format(type(x)))

    if x.shape[axis] < frame_length:
        raise Exception('Input is too short (n={:d})'
                        ' for frame_length={:d}'.format(x.shape[axis], frame_length))

    if hop_length < 1:
        raise Exception('Invalid hop_length: {:d}'.format(hop_length))

    if axis == -1 and not x.flags['F_CONTIGUOUS']:
        x = np.asfortranarray(x)
    elif axis == 0 and not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)

    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    strides = np.asarray(x.strides)

    new_stride = np.prod(strides[strides > 0] // x.itemsize) * x.itemsize

    if axis == -1:
        shape = list(x.shape)[:-1] + [frame_length, n_frames]
        strides = list(strides) + [hop_length * new_stride]

    elif axis == 0:
        shape = [n_frames, frame_length] + list(x.shape)[1:]
        strides = [hop_length * new_stride] + list(strides)

    else:
        raise Exception('Frame axis={} must be either 0 or -1'.format(axis))

    return as_strided(x, shape=shape, strides=strides)


def stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann',
         center=True, dtype=np.complex64, pad_mode='reflect'):
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, n_fft)

    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))

    # Pad the time series so that frames are centered
    if center:
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)

    # Window the time series.
    y_frames = frame(y, frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    stft_matrix = np.empty((int(1 + n_fft // 2), y_frames.shape[1]),
                           dtype=dtype,
                           order='F')

    # how many columns can we fit within MAX_MEM_BLOCK?
    n_columns = int(MAX_MEM_BLOCK / (stft_matrix.shape[0] *
                                     stft_matrix.itemsize))

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        # RFFT and Conjugate here to match phase from DPWE code
        stft_matrix[:, bl_s:bl_t] = fft.fft(fft_window *
                                            y_frames[:, bl_s:bl_t],
                                            axis=0)[:stft_matrix.shape[0]].conj()

    return stft_matrix


def magphase(d, power=1):
    mag = np.abs(d)
    mag **= power
    phase = np.exp(1.j * np.angle(d))

    return mag, phase
