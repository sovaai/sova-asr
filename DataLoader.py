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


def preprocess(audio_path, sample_rate=16000, window_size=0.02, window_stride=0.01, window='hamming'):
    audio = load_audio(audio_path, sample_rate)
    nfft = int(sample_rate * window_size)
    win_length = nfft
    hop_length = int(sample_rate * window_stride)

    d = stft(audio, n_fft=nfft, hop_length=hop_length,
             win_length=win_length, window=window)

    spect, phase = magphase(d)
    pcen_result = pcen2(e=spect, sr=sample_rate, hop_length=hop_length)
    mean_pcen = pcen_result.mean()
    std_pcen = pcen_result.std()

    pcen_result = np.add(pcen_result, -mean_pcen)
    pcen_result = pcen_result / std_pcen

    return pcen_result


def get_batch(batch):
    longest_sample = max(batch, key=lambda p: p[0].shape[1])[0]
    freq_size = longest_sample.shape[0]
    mini_batch_size = len(batch)
    max_seq_length = longest_sample.shape[1]
    inputs = np.zeros((mini_batch_size, freq_size, max_seq_length))
    target_sizes = np.zeros(shape=(mini_batch_size,), dtype=int)
    input_percentages = np.zeros(shape=(mini_batch_size,), dtype=float)
    targets = []
    input_file_path_and_transcription = []

    for x in range(mini_batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        tensor_path = sample[2]
        original_transcription = sample[3]
        seq_length = tensor.shape[1]
        tensor_new = np.pad(tensor, ((0, 0), (0, abs(seq_length - max_seq_length))), 'wrap')
        inputs[x] = tensor_new
        input_percentages[x] = seq_length / float(max_seq_length)
        target_sizes[x] = len(target)
        targets.extend(target)
        input_file_path_and_transcription.append([tensor_path, original_transcription])

    targets = np.array(targets)

    return inputs, input_percentages, targets, target_sizes, input_file_path_and_transcription


class DataLoader(object):
    def __init__(self, dataset, batch_sampler):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.sample_iter = iter(self.batch_sampler)

    def __next__(self):
        try:
            indices = next(self.sample_iter)
            indices = [i for i in indices][0]
            batch = get_batch([self.dataset[i] for i in indices])
            return batch
        except Exception as e:
            print("Encountered exception {}".format(e))
            raise StopIteration()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.batch_sampler)

    def reset(self):
        self.batch_sampler.reset()


class SpectrogramDataset(object):
    def __init__(self, labels, sample_rate, window_size, window_stride, manifest_file_path):
        self.manifest_file_path = manifest_file_path
        with open(self.manifest_file_path) as f:
            lines = f.readlines()
        self.ids = [x.strip().split(',') for x in lines]
        self.size = len(lines)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, transcript_loaded = sample[0], sample[1]
        spectrogram = preprocess(audio_path, self.sample_rate, self.window_size, self.window_stride)
        transcript = list(filter(None, [self.labels_map.get(x) for x in list(transcript_loaded)]))
        return spectrogram, transcript, audio_path, transcript_loaded

    def __len__(self):
        return self.size


class BucketingSampler(object):
    def __init__(self, data_source, batch_size=1, shuffle=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.ids = list(range(0, len(data_source)))
        self.batch_id = 0
        self.bins = []
        self.shuffle = shuffle
        self.reset()

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_id < len(self):
            ids = self.bins[self.batch_id]
            self.batch_id += 1
            yield ids
        else:
            raise StopIteration()

    def __len__(self):
        return len(self.bins)

    def get_bins(self):
        if self.shuffle:
            np.random.shuffle(self.ids)
        self.bins = [self.ids[i:i + self.batch_size] for i in range(0, len(self.ids), self.batch_size)]

    def reset(self):
        self.get_bins()
        self.batch_id = 0
