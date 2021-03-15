import numpy as np
import math
import itertools
from scipy.special import softmax
np.seterr(divide='ignore')


class GreedyDecoder:
    def __init__(self, labels, blank=0):
        self.labels = labels
        self.blank = blank

    def decode(self, output, max_len=None):
        output = softmax(output.astype(np.float32), axis=-1)
        best_path = np.argmax(output, axis=1)
        if max_len is not None:
            best_path = best_path[:max_len]
        return "".join(self.labels[k] for k, _ in itertools.groupby(best_path) if k != self.blank)


class TrieDecoder:
    def __init__(self, lexicon, tokens, lm_path, beam_threshold=30):
        from TrieDecoder.Common import Dictionary, create_word_dict, load_words
        from TrieDecoder.Decoder import CriterionType, DecoderOptions, KenLM, LexiconDecoder
        lexicon = load_words(lexicon)
        self.wordDict = create_word_dict(lexicon)
        self.tokenDict = Dictionary(tokens)
        self.lm = KenLM(lm_path, self.wordDict)

        trie, self.sil_idx, self.blank_idx, self.unk_idx = self.get_trie(lexicon)
        transitions = np.zeros((self.tokenDict.index_size(), self.tokenDict.index_size())).flatten()

        opts = DecoderOptions(
            2000, 100, beam_threshold, 1.4, 1.0, -math.inf, -1, 0, False, CriterionType.CTC
        )

        self.trieDecoder = LexiconDecoder(
            opts, trie, self.lm, self.sil_idx, self.blank_idx, self.unk_idx, transitions, False
        )

    def get_trie(self, lexicon):
        from TrieDecoder.Common import tkn_to_idx
        from TrieDecoder.Decoder import SmearingMode, Trie
        sil_idx = self.tokenDict.get_index("|")
        unk_idx = self.wordDict.get_index("<unk>")
        blank_idx = self.tokenDict.get_index("#")

        trie = Trie(self.tokenDict.index_size(), sil_idx)
        start_state = self.lm.start(False)

        for word, spellings in lexicon.items():
            usr_idx = self.wordDict.get_index(word)
            _, score = self.lm.score(start_state, usr_idx)
            for spelling in spellings:
                spelling_indices = tkn_to_idx(spelling, self.tokenDict, 0)
                trie.insert(spelling_indices, usr_idx, score)

        trie.smear(SmearingMode.MAX)

        return trie, sil_idx, blank_idx, unk_idx

    def process_string(self, sequence):
        string = ''
        for i in range(len(sequence)):
            char = self.tokenDict.get_entry(sequence[i])
            if char != self.tokenDict.get_entry(self.blank_idx):
                if i != 0 and char == self.tokenDict.get_entry(sequence[i - 1]):
                    pass
                elif char == self.tokenDict.get_entry(self.sil_idx):
                    string += ' '
                else:
                    string = string + char
        return ' '.join(string.split())

    def decode(self, output):
        output = softmax(output[:, :].astype(np.float32), axis=-1)
        t, n = output.shape
        emissions = np.log(output)
        result = self.trieDecoder.decode(emissions.ctypes.data, t, n)[0]

        return self.process_string(result.tokens)
