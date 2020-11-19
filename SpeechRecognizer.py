import numpy as np
import argparse
import configparser
from DataLoader import DataLoader
from Decoder import GreedyDecoder


class SpeechRecognizer(object):
    def __init__(self, config_path='config.ini'):
        if config_path is None:
            raise Exception('Path to config file is None')
        self.config = configparser.ConfigParser()
        self.config.read(config_path, encoding='UTF-8')
        self.labels = self.config['Wav2Letter']['labels'][1:-1]
        self.sample_rate = int(self.config['Wav2Letter']['sample_rate'])
        self.window_size = float(self.config['Wav2Letter']['window_size'])
        self.window_stride = float(self.config['Wav2Letter']['window_stride'])
        self.greedy = int(self.config['Wav2Letter']['greedy'])
        self.cpu = int(self.config['Wav2Letter']['cpu'])

        if self.cpu:
            from PuzzleLib import Config
            Config.backend = Config.Backend.cpu

        from PuzzleLib.Models.Nets.WaveToLetter import loadW2L
        from PuzzleLib.Modules import MoveAxis

        self.data_loader = DataLoader(
            sample_rate=self.sample_rate,
            window_size=self.window_size,
            window_stride=self.window_stride
        )

        nfft = int(self.sample_rate * self.window_size)
        self.w2l = loadW2L(modelpath=self.config['Wav2Letter']['model_path'], inmaps=(1 + nfft // 2),
                           nlabels=len(self.labels))
        self.w2l.append(MoveAxis(src=2, dst=0))

        if not self.cpu:
            self.w2l.calcMode(np.float16)

        self.w2l.evalMode()

        if not self.greedy:
            from Decoder import TrieDecoder
            lexicon = self.config['Wav2Letter']['lexicon']
            tokens = self.config['Wav2Letter']['tokens']
            lm_path = self.config['Wav2Letter']['lm_path']
            beam_threshold = float(self.config['Wav2Letter']['beam_threshold'])
            self.decoder = TrieDecoder(lexicon, tokens, lm_path, beam_threshold)
        else:
            self.decoder = GreedyDecoder(self.labels)

    def recognize(self, audio_path):
        preprocessed_audio = self.data_loader.preprocess(audio_path)
        if self.cpu:
            from PuzzleLib.CPU.CPUArray import CPUArray
            inputs = CPUArray.toDevice(np.array([preprocessed_audio]).astype(np.float32))
        else:
            from PuzzleLib.Backend import gpuarray
            inputs = gpuarray.to_gpu(np.array([preprocessed_audio]).astype(np.float16))

        output = self.w2l(inputs).get()
        output = np.vstack(output).astype(np.float32)
        result = self.decoder.decode(output)

        if not self.cpu:
            from PuzzleLib.Backend.gpuarray import memoryPool
            memoryPool.freeHeld()

        del inputs, output

        return [result]


def test():
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('--audio', default='Data/test.wav', metavar='DIR', help='Path to wav file')
    parser.add_argument('--config', default='config.ini', help='Path to config')
    args = parser.parse_args()

    recognizer = SpeechRecognizer(args.config)

    print(recognizer.recognize(args.audio))


if __name__ == "__main__":
    test()
