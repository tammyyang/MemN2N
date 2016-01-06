import lasagne
import numpy as np
import sys
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from simdat.core import ml
import warnings
warnings.filterwarnings('ignore', '.*topo.*')

class Pred:
    def __init__(self):
        self.ml = ml.MLTools()
        self.network = None

    def get_lines(self, fname):
        lines = []
        with open(fname, 'r') as f:
            ori_lines = f.readlines()
            for line in ori_lines:
                line = line.strip()
                first_break = line.find(' ')
                _line = line[first_break+1:]
                if _line.find('?') == -1:
                    lines.append({'type': 's', 'text': _line})
                else:
                    sid = int(line[0:first_break])
                    qbreak = _line.find('?')
                    ans = _line[qbreak+1:].replace(' ', '')
                    tmp = [w for w in ans.split('\t') if len(w) > 0]
                    lines.append({'id': sid, 'type': 'q',
                                  'text': _line[:qbreak+1],
                                  'answer': tmp[0],
                                  'refs': [int(tmp[1])-1]})
                if False and i > 1000:
                    break
        return np.array(lines)

    def pred(self, data, network, fmodel='rnn.pkl'):
        prev_weights = self.ml.read_model(fmodel)
        lasagne.layers.helper.set_all_param_values(network, prev_weights)

def main():
    pred = Pred()
    data = pred.get_lines('tmp.txt')
    pred.pred(data)

if __name__ == '__main__':
    main()
