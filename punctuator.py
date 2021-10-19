import os
import math
import numpy as np
from bert_punctuator.bert import BertPunc, BertConfig
from bert_punctuator.tokenizer import BertTokenizer
from PuzzleLib.Backend import gpuarray
from PuzzleLib.Config import getLogger
import logging


logger = getLogger()
logger.setLevel(logging.ERROR)


class Punctuator(object):
    def __init__(self, model_path="data/punctuator"):
        self.tokenizer = BertTokenizer(os.path.join(model_path, "vocab.txt"), lower_case=True)
        
        conf = BertConfig(os.path.join(model_path, "config.json"))
        self.segment_size = conf.segment_size

        self.punctuation_enc = {
            " ": 0,
            ", ": 1,
            ". ": 2,
            "? ": 3
        }
        self.punctuation_dec = {i:key for key, i in self.punctuation_enc.items()}
        
        self.bert_punctuator = BertPunc(conf)
        self.bert_punctuator.evalMode()
        self.bert_punctuator.calcMode(np.float16)
        self.bert_punctuator.load(os.path.join(model_path, "bert16.hdf"))
        
        
    def segment(self, ids):
        x = []
        x_pad = ids[-((self.segment_size - 1) // 2 - 1):] + ids + ids[:self.segment_size // 2]

        for i in range(len(x_pad) - self.segment_size + 2):
            segment = x_pad[i:i + self.segment_size - 1]
            segment.insert((self.segment_size - 1) // 2, 0)
            x.append(segment)

        return np.array(x)

    
    def preprocess_data(self, txt):
        data =  txt.split()
        token_count = []
        x = []
        for word in data:
            tokens = self.tokenizer.tokenize(word)
            ids = self.tokenizer.convert_tokens_to_ids(tokens)
            if len(ids) > 0:
                x += ids
                token_count.append([word, len(ids)])
        x = self.segment(x)
        return x, token_count
    
    
    def get_predictions(self, batches):
        y_pred = []
        for batch in batches:
            inputs = gpuarray.to_gpu(batch.astype(np.int32))
            output = self.bert_punctuator(inputs).get()
            y_pred += list(output.argmax(axis=1).flatten())
        return y_pred
    
    
    def convert_predictions(self, token_count, y):
        i = 0
        s = ""
        for word, k in token_count:
            i += k
            punc = self.punctuation_dec[y[i - 1]]
            if i == len(y) and punc not in [". ", "? "]:
                punc = ". "
            s = s + word + punc
      
        pred = s[0].upper() + s[1]
        for i in range(2, len(s)):
            if s[i - 2] in [".", "?"]:
                pred += s[i].upper()
            else:
                pred += s[i]
        
        return pred[:-1]
    
    
    def predict(self, txt):
        txt = "берт расставляет знаки препинания в строке предсказывая токены знаков препинания. " + txt
        x, token_count = self.preprocess_data(txt)
        
        batches = np.array_split(x, math.ceil(x.shape[0] / 8), axis=0)
        
        y_pred = self.get_predictions(batches)

        pred = self.convert_predictions(token_count, y_pred)[83:]

        return pred
