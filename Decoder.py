import numpy as np
import math
import itertools
from scipy.special import softmax
np.seterr(divide='ignore')


class DecodeResult:
    def __init__(self, score, words):
        self.score, self.words = score, words
        self.text = " ".join(word["word"] for word in words)


class GreedyDecoder:
    def __init__(self, labels, blank_idx=0):
        self.labels, self.blank_idx = labels, blank_idx
        self.delim_idx = self.labels.index("|")


    def decode(self, output, start_timestamp=0, frame_time=0.02):
        best_path = np.argmax(output.astype(np.float32, copy=False), axis=1)
        score = None

        words, new_word, i = [], True, 0
        current_word, current_timestamp, end_idx = None, start_timestamp, 0
        words_len = 0

        for k, g in itertools.groupby(best_path):
            if k != self.blank_idx:
                if new_word and k != self.delim_idx:
                    new_word, start_idx = False, i
                    current_word, current_timestamp = self.labels[k], frame_time * i + start_timestamp

                elif k == self.delim_idx:
                    end_timestamp = frame_time * i + start_timestamp
                    new_word, end_idx = True, i
                    word_score = output[range(start_idx, end_idx), best_path[range(start_idx, end_idx)]] - np.max(output)
                    if score is not None:
                        score = np.hstack([score, word_score])
                    else:
                        score = word_score
                    word_confidence = np.round(np.exp(word_score.mean() / max(1, end_idx - start_idx)) * 100.0, 2)
                    words_len += end_idx - start_idx
                    words.append({
                        "word": current_word,
                        "start": np.round(current_timestamp, 2),
                        "end": np.round(end_timestamp, 2),
                        "confidence": word_confidence
                    })

                else:
                    current_word += self.labels[k]

            i += sum(1 for _ in g)

        score = np.round(np.exp(score.mean() / max(1, words_len)) * 100.0, 2)

        return DecodeResult(score, words)


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
        self.delim_idx = self.tokenDict.get_index("|")

    def get_trie(self, lexicon):
        from TrieDecoder.Common import tkn_to_idx
        from TrieDecoder.Decoder import SmearingMode, Trie
        unk_idx = self.wordDict.get_index("<unk>")
        sil_idx = blank_idx = self.tokenDict.get_index("#")

        trie = Trie(self.tokenDict.index_size(), sil_idx)
        start_state = self.lm.start(False)

        for word, spellings in lexicon.items():
            usr_idx = self.wordDict.get_index(word)
            _, score = self.lm.score(start_state, usr_idx)
            score = np.round(score, 2)

            for spelling in spellings:
                spelling_indices = tkn_to_idx(spelling, self.tokenDict, 0)
                trie.insert(spelling_indices, usr_idx, score)

        trie.smear(SmearingMode.MAX)

        return trie, sil_idx, blank_idx, unk_idx

    def decode(self, output, start_timestamp=0, frame_time=0.02):
        output = np.log(softmax(output[:, :].astype(np.float32, copy=False), axis=-1))

        t, n = output.shape
        result = self.trieDecoder.decode(output.ctypes.data, t, n)[0]
        tokens = result.tokens

        words, new_word = [], True
        current_word, current_timestamp, start_idx, end_idx = None, start_timestamp, 0, 0
        lm_state = self.lm.start(False)
        words_len = 0

        for i, k in enumerate(tokens):
            if k != self.blank_idx:
                if i > 0 and k == tokens[i - 1]:
                    pass

                elif k == self.sil_idx:
                    new_word = True

                else:
                    if new_word and k != self.delim_idx:
                        new_word = False
                        current_word, current_timestamp = self.tokenDict.get_entry(k), frame_time * i + start_timestamp
                        start_idx = i

                    elif k == self.delim_idx:
                        new_word, end_idx = True, i
                        lm_state, word_lm_score = self.lm.score(lm_state, self.wordDict.get_index(current_word))
                        end_timestamp = frame_time * i + start_timestamp
                        words_len += end_idx - start_idx
                        words.append({
                            "word": current_word,
                            "start": np.round(current_timestamp, 2),
                            "end": np.round(end_timestamp, 2),
                            "confidence": np.round(np.exp(word_lm_score / max(1, end_idx - start_idx)) * 100, 2)
                        })

                    else:
                        current_word += self.tokenDict.get_entry(k)

        score = np.round(np.exp(result.score / max(1, words_len)), 2)

        return DecodeResult(score, words)
