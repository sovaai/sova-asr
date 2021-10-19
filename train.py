import argparse
import numpy as np
import os
from shutil import copyfile
import configparser
import Levenshtein
from data_loader import DataLoader, SpectrogramDataset, BucketingSampler
from decoder import GreedyDecoder
from PuzzleLib.Models.Nets.WaveToLetter import loadW2L
from PuzzleLib.Backend import gpuarray
from PuzzleLib.Cost.CTC import CTC
from PuzzleLib.Optimizers.Adam import Adam
from PuzzleLib.Modules import MoveAxis
from PuzzleLib.Modules.Cast import Cast


def get_data_loader(manifest_file_path, labels, sample_rate, window_size, window_stride, batch_size):
    dataset = SpectrogramDataset(labels, sample_rate, window_size, window_stride, manifest_file_path)
    sampler = BucketingSampler(dataset, batch_size=batch_size)
    return DataLoader(dataset, batch_sampler=sampler)


def calculate_wer(s1, s2):
    b = set(s1.split() + s2.split())
    word2char = dict(zip(b, range(len(b))))
    w1 = [chr(word2char[w]) for w in s1.split()]
    w2 = [chr(word2char[w]) for w in s2.split()]
    return Levenshtein.distance(''.join(w1), ''.join(w2)) / len(''.join(w2))


def calculate_cer(s1, s2):
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Levenshtein.distance(s1, s2) / len(s2)


def train(model, ctc, optimizer, loader, checkpoint_per_batch, save_folder, save_name, fp16):
    model.reset()
    model.trainMode()
    loader.reset()

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    for i, (data) in enumerate(loader):
        inputs, input_percentages, targets, target_sizes, _ = data
        if fp16:
            inputs = gpuarray.to_gpu(inputs.astype(np.float16))
        else:
            inputs = gpuarray.to_gpu(inputs.astype(np.float32))

        out = model(inputs)
        out_len = gpuarray.to_gpu((out.shape[0] * input_percentages).astype(np.int32))
        target_sizes = target_sizes.astype(np.int32)
        targets = gpuarray.to_gpu(targets.astype(np.int32))

        error, grad = ctc([out, out_len], [targets, target_sizes])

        print('Training iter {} of {}, CTC: {}'.format(i + 1, len(loader), error))

        optimizer.zeroGradParams()
        model.backward(grad.astype(np.float32), updGrad=False)
        optimizer.update()

        if checkpoint_per_batch and i % checkpoint_per_batch == 0 and i > 0:
            save_path = os.path.join(save_folder, '{}_iter_{}.hdf'.format(save_name, i))
            model.save(hdf=save_path)
            copyfile(save_path, os.path.join(save_folder, 'last.hdf'))

    save_path = os.path.join(save_folder, '{}.hdf'.format(save_name))
    model.save(hdf=save_path)
    copyfile(save_path, os.path.join(save_folder, 'last.hdf'))

    return model


def validate(model, loader, decoder, fp16):
    loader.reset()
    model.evalMode()
    total_cer, total_wer = 0, 0

    for i, (data) in enumerate(loader):
        inputs, input_percentages, targets, target_sizes, input_file = data
        if fp16:
            inputs = gpuarray.to_gpu(inputs.astype(np.float16))
        else:
            inputs = gpuarray.to_gpu(inputs.astype(np.float32))

        out = model(inputs)
        out_len = (out.shape[0] * input_percentages).astype(np.int32)

        decoded_output = [
            decoder.decode(output, max_len=out_len[j]) for j, output in enumerate(np.moveaxis(out.get(), 0, 1))
        ]

        print('\nValidation iter {} of {}'.format(i + 1, len(loader)))

        wer, cer = 0, 0
        for x in range(len(decoded_output)):
            transcript, reference = decoded_output[x], input_file[x][1]
            print('Transcript: {}\nReference: {}\nFilepath: {}'.format(transcript, reference, input_file[x][0]))
            try:
                wer += calculate_wer(transcript, reference)
                cer += calculate_cer(transcript, reference)
            except Exception as e:
                print('Encountered exception {}'.format(e))
        total_cer += cer
        total_wer += wer

    wer = total_wer / len(loader.dataset) * 100
    cer = total_cer / len(loader.dataset) * 100
    print('WER: {}'.format(wer))
    print('CER: {}'.format(cer))


def main():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', metavar='DIR', help='Path to train config', default='config.ini')
    args = parser.parse_args()

    config_path = args.config
    if config_path is None:
        raise Exception('Path to config file is None')
    config = configparser.ConfigParser()
    config.read(config_path, encoding='UTF-8')

    sample_rate = int(config['Wav2Letter'].get('sample_rate'))
    window_size = float(config['Wav2Letter'].get('window_size'))
    window_stride = float(config['Wav2Letter'].get('window_stride'))
    labels = config['Wav2Letter'].get('labels')[1:-1]

    train_manifest = config['Train'].get('train_manifest', None)
    val_manifest = config['Train'].get('val_manifest', None)
    epochs = int(config['Train'].get('epochs'))
    batch_size = int(config['Train'].get('batch_size'))
    learning_rate = float(config['Train'].get('learning_rate'))
    fp16 = bool(config['Train'].get('fp16'))
    checkpoint_name = config['Train'].get('checkpoint_name')
    checkpoint_per_batch = int(config['Train'].get('checkpoint_per_batch'))
    save_folder = config['Train'].get('save_folder')
    continue_from = config['Train'].get('continue_from')

    train_loader, val_loader = None, None

    if train_manifest is not None:
        train_loader = get_data_loader(train_manifest, labels, sample_rate, window_size, window_stride, batch_size)

    if val_manifest is not None:
        val_loader = get_data_loader(val_manifest, labels, sample_rate, window_size, window_stride, batch_size)

    nfft = int(sample_rate * window_size)
    w2l = loadW2L(modelpath=continue_from, inmaps=(1 + nfft // 2), nlabels=len(labels))

    if fp16:
        w2l.calcMode(np.float16)
        w2l.append(Cast(np.float16, np.float32))

    w2l.append(MoveAxis(src=2, dst=0))

    blank_index = [i for i in range(len(labels)) if labels[i] == '_'][0]
    ctc = CTC(blank_index)

    adam = Adam(alpha=learning_rate)
    adam.setupOn(w2l, useGlobalState=True)

    decoder = GreedyDecoder(labels, blank_index)

    for epoch in range(epochs):
        if train_manifest is not None:
            print('Epoch {} of {}'.format(epoch + 1, epochs))
            w2l = train(w2l, ctc, adam, train_loader, checkpoint_per_batch, save_folder,
                        '{}_{}'.format(checkpoint_name, epoch), fp16)

        if val_manifest is not None:
            validate(w2l, val_loader, decoder, fp16)


if __name__ == '__main__':
    main()
