import numpy as np
import scipy
from Utils import stft, magphase
from pydub import AudioSegment


def pcen2(e, sr=16000, hop_length=512, t=0.395, eps=0.000001, alpha=0.98, delta=2.0, r=0.5):
    s = 1 - np.exp(-float(hop_length) / (t * sr))
    m = scipy.signal.lfilter([s], [1, s - 1], e)
    smooth = (eps + m) ** (-alpha)

    return (e * smooth + delta) ** r - delta ** r


def load_audio(path, sample_rate):
    sound = AudioSegment.from_wav(path)
    sound = sound.set_frame_rate(sample_rate)
    sound = sound.set_channels(1)
    sound = sound.set_sample_width(2)

    return np.array(sound.get_array_of_samples()).astype(float)


class DataLoader(object):
    def __init__(self, sample_rate=16000, window_size=0.02, window_stride=0.01):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride

    def preprocess(self, audio_path, window='hamming'):
        audio = load_audio(audio_path, self.sample_rate)
        nfft = int(self.sample_rate * self.window_size)
        win_length = nfft
        hop_length = int(self.sample_rate * self.window_stride)

        d = stft(audio, n_fft=nfft, hop_length=hop_length,
                 win_length=win_length, window=window)

        spect, phase = magphase(d)
        pcen_result = pcen2(e=spect, sr=self.sample_rate, hop_length=hop_length)
        mean_pcen = pcen_result.mean()
        std_pcen = pcen_result.std()

        pcen_result = np.add(pcen_result, -mean_pcen)
        pcen_result = pcen_result / std_pcen

        return pcen_result
