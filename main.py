from __future__ import division
import argparse
import glob
import lasagne
import nltk
import numpy as np
import sys
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from simdat.core import ml
from data import read_data_qa
from data import process_dataset
from theano.printing import Print as pp

import warnings
warnings.filterwarnings('ignore', '.*topo.*')

class InnerProductLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, nonlinearity=None, **kwargs):
        super(InnerProductLayer, self).__init__(incomings, **kwargs)
        self.nonlinearity = nonlinearity
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0][:2]

    def get_output_for(self, inputs, **kwargs):
        M = inputs[0]
        u = inputs[1]
        output = T.batched_dot(M, u)
        if self.nonlinearity is not None:
            output = self.nonlinearity(output)
        return output


class BatchedDotLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(BatchedDotLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 2:
            raise NotImplementedError

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2])

    def get_output_for(self, inputs, **kwargs):
        return T.batched_dot(inputs[0], inputs[1])


class SumLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axis, **kwargs):
        super(SumLayer, self).__init__(incoming, **kwargs)
        self.axis = axis

    def get_output_shape_for(self, input_shape):
        return input_shape[:self.axis] + input_shape[self.axis+1:]

    def get_output_for(self, input, **kwargs):
        return T.sum(input, axis=self.axis)


class TemporalEncodingLayer(lasagne.layers.Layer):

    def __init__(self, incoming, T=lasagne.init.Normal(std=0.1), **kwargs):
        super(TemporalEncodingLayer, self).__init__(incoming, **kwargs)
        self.T = self.add_param(T, self.input_shape[-2:], name="T")

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, **kwargs):
        return input + self.T


class TransposedDenseLayer(lasagne.layers.DenseLayer):

    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(TransposedDenseLayer, self).__init__(incoming, num_units, W, b, nonlinearity, **kwargs)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            input = input.flatten(2)

        activation = T.dot(input, self.W.T)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)


class MemoryNetworkLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, vocab, embedding_size, A, A_T, C, C_T, nonlinearity=lasagne.nonlinearities.softmax, **kwargs):
        super(MemoryNetworkLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 3:
            raise NotImplementedError

        batch_size, maxseq, maxsent = self.input_shapes[0]

        l_context_in = lasagne.layers.InputLayer(shape=(batch_size, maxseq, maxsent))
        l_B_embedding = lasagne.layers.InputLayer(shape=(batch_size, embedding_size))
        l_context_pe_in = lasagne.layers.InputLayer(shape=(batch_size, maxseq, maxsent, embedding_size))

        l_context_in = lasagne.layers.ReshapeLayer(l_context_in, shape=(batch_size * maxseq * maxsent, ))
        l_A_embedding = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=A)
        self.A = l_A_embedding.W
        l_A_embedding = lasagne.layers.ReshapeLayer(l_A_embedding, shape=(batch_size, maxseq, maxsent, embedding_size))
        l_A_embedding = lasagne.layers.ElemwiseMergeLayer((l_A_embedding, l_context_pe_in), merge_function=T.mul)
        l_A_embedding = SumLayer(l_A_embedding, axis=2)
        l_A_embedding = TemporalEncodingLayer(l_A_embedding, T=A_T)
        self.A_T = l_A_embedding.T

        l_C_embedding = lasagne.layers.EmbeddingLayer(l_context_in, len(vocab)+1, embedding_size, W=C)
        self.C = l_C_embedding.W
        l_C_embedding = lasagne.layers.ReshapeLayer(l_C_embedding, shape=(batch_size, maxseq, maxsent, embedding_size))
        l_C_embedding = lasagne.layers.ElemwiseMergeLayer((l_C_embedding, l_context_pe_in), merge_function=T.mul)
        l_C_embedding = SumLayer(l_C_embedding, axis=2)
        l_C_embedding = TemporalEncodingLayer(l_C_embedding, T=C_T)
        self.C_T = l_C_embedding.T

        l_prob = InnerProductLayer((l_A_embedding, l_B_embedding), nonlinearity=nonlinearity)
        l_weighted_output = BatchedDotLayer((l_prob, l_C_embedding))

        l_sum = lasagne.layers.ElemwiseSumLayer((l_weighted_output, l_B_embedding))

        self.l_context_in = l_context_in
        self.l_B_embedding = l_B_embedding
        self.l_context_pe_in = l_context_pe_in
        self.network = l_sum

        params = lasagne.layers.helper.get_all_params(self.network, trainable=True)
        values = lasagne.layers.helper.get_all_param_values(self.network, trainable=True)
        for p, v in zip(params, values):
            self.add_param(p, v.shape, name=p.name)

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size, dtype=theano.config.floatX)
        self.set_zero = theano.function([zero_vec_tensor], updates=[(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [self.A, self.C]])

    def get_output_shape_for(self, input_shapes):
        return lasagne.layers.helper.get_output_shape(self.network)

    def get_output_for(self, inputs, **kwargs):
        return lasagne.layers.helper.get_output(self.network, {self.l_context_in: inputs[0], self.l_B_embedding: inputs[1], self.l_context_pe_in: inputs[2]})

    def reset_zero(self):
        self.set_zero(self.zero_vec)


class RNNArgs(ml.Args):
    def _add_args(self):
        """Called by __init__ of Args class"""
        self.trainf = 'data/en/qa1_single-supporting-fact_train.txt'
        self.validf = 'data/en/qa1_single-supporting-fact_test.txt'
        self.batch_size = 32
        self.embedding_size = 20
        self.max_norm = 40
        self.lr = 0.01
        self.linear_start = True


class Model:
    def __init__(self, trainf, validf, batch_size=32,
                 embedding_size=20, max_norm=40, lr=0.01,
                 num_hops=3, adj_weight_tying=True,
                 linear_start=True, **kwargs):
        self.ml = ml.MLTools()
        self.args = RNNArgs(pfs=['rnn.json'])
        self.count = []
        self.S = []
        self.word2idx = {}
        self.idx2word = {}
        self.data = {'train': {}, 'valid': {}}

        maxseq, maxsent, ptrain = self.init_data(self.args.trainf)
        self.data['train']['C'] = ptrain['C']
        self.data['train']['Q'] = ptrain['Q']
        self.data['train']['Y'] = ptrain['Y']

        maxseq, maxsent, pvalid = self.init_data(self.args.validf,
                                                maxseq=maxseq,
                                                maxsent=maxsent)
        self.data['valid']['C'] = pvalid['C']
        self.data['valid']['Q'] = pvalid['Q']
        self.data['valid']['Y'] = pvalid['Y']
        self.S = np.array(self.S, dtype=np.int32)

        self.idx2word = dict(zip(self.word2idx.values(), self.word2idx.keys()))
        vocab = self.word2idx.keys()

        print 'batch_size:', batch_size, 'maxseq:', maxseq, 'maxsent:', maxsent
        print 'sentences:', self.S.shape
        print 'vocab:', len(vocab), vocab
        for d in ['train', 'valid']:
            print d,
            for k in ['C', 'Q', 'Y']:
                print k, self.data[d][k].shape,
            print ''

        lb = LabelBinarizer()
        lb.fit(list(vocab))
        vocab = lb.classes_.tolist()

        self.batch_size = batch_size
        self.maxseq = maxseq
        self.maxsent = maxsent
        self.embedding_size = embedding_size
        self.num_classes = len(vocab) + 1
        self.vocab = vocab
        self.adj_weight_tying = adj_weight_tying
        self.num_hops = num_hops
        self.lb = lb
        self.init_lr = lr
        self.lr = self.init_lr
        self.max_norm = max_norm
        #TAMMY
        self.nonlinearity = None if linear_start else lasagne.nonlinearities.softmax

        self.build_network(self.nonlinearity)

    def init_data(self, fdata, maxseq=0, maxsent=0):

        data, maxseq, maxsent = read_data_qa(fdata, self.count,
                                             self.word2idx, maxseq, maxsent)
        processed = process_dataset(data, self.word2idx, maxsent,
                                    offset=len(self.S))
        self.S.extend(processed['S'])
        return maxseq, maxsent, processed

    def build_network(self, nonlinearity):
        batch_size = self.batch_size
        maxseq, maxsent = self.maxseq, self.maxsent
        embedding_size, vocab = self.embedding_size, self.vocab

        c = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        c_pe = T.tensor4()
        q_pe = T.tensor4()

        _shape = (batch_size, maxseq)
        self.c_shared = theano.shared(np.zeros(_shape, dtype=np.int32),
                                      borrow=True)

        _shape = (batch_size, )
        self.q_shared = theano.shared(np.zeros(_shape, dtype=np.int32),
                                      borrow=True)

        _shape = (batch_size, self.num_classes)
        self.a_shared = theano.shared(np.zeros(_shape, dtype=np.int32),
                                      borrow=True)

        _shape = (batch_size, maxseq, maxsent, embedding_size)
        self.c_pe_shared = theano.shared(np.zeros(_shape,
                                                  dtype=theano.config.floatX),
                                         borrow=True)

        _shape = (batch_size, 1, maxsent, embedding_size)
        self.q_pe_shared = theano.shared(np.zeros(_shape,
                                                  dtype=theano.config.floatX),
                                         borrow=True)

        S_shared = theano.shared(self.S, borrow=True)

        _shape = (batch_size, maxseq, maxsent)
        cc = S_shared[c.flatten()].reshape(_shape)

        _shape = (batch_size, maxsent)
        qq = S_shared[q.flatten()].reshape(_shape)

        _shape = (batch_size, maxseq, maxsent)
        l_context_in = lasagne.layers.InputLayer(shape=_shape)

        _shape = (batch_size, maxsent)
        l_question_in = lasagne.layers.InputLayer(shape=_shape)

        _shape = (batch_size, maxseq, maxsent, embedding_size)
        l_context_pe_in = lasagne.layers.InputLayer(shape=_shape)

        _shape = (batch_size, 1, maxsent, embedding_size)
        l_question_pe_in = lasagne.layers.InputLayer(shape=_shape)

        std = {'std': 0.1}
        A = lasagne.init.Normal(**std).sample((len(vocab)+1, embedding_size))
        C = lasagne.init.Normal(**std)
        A_T = lasagne.init.Normal(**std)
        C_T = lasagne.init.Normal(**std)
        W = A if self.adj_weight_tying else lasagne.init.Normal(**std)

        _shape = (batch_size * maxsent, )
        l_question_in = lasagne.layers.ReshapeLayer(l_question_in,
                                                    shape=_shape)
        l_B_embedding = lasagne.layers.EmbeddingLayer(l_question_in,
                                                      len(vocab)+1,
                                                      embedding_size, W=W)
        B = l_B_embedding.W
        _shape = (batch_size, 1, maxsent, embedding_size)
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding,
                                                    shape=_shape)
        l_B_embedding = lasagne.layers.ElemwiseMergeLayer((l_B_embedding,
                                                           l_question_pe_in),
                                                          merge_function=T.mul)
        _shape = (batch_size, maxsent, embedding_size)
        l_B_embedding = lasagne.layers.ReshapeLayer(l_B_embedding,
                                                    shape=_shape)
        l_B_embedding = SumLayer(l_B_embedding, axis=1)

        self.mem_layers = [MemoryNetworkLayer((l_context_in, l_B_embedding,
                                               l_context_pe_in),
                                              vocab, embedding_size,
                                              A=A, A_T=A_T, C=C, C_T=C_T,
                                              nonlinearity=nonlinearity)]
        for _ in range(1, self.num_hops):
            if self.adj_weight_tying:
                A = self.mem_layers[-1].C
                C = lasagne.init.Normal(**std)
                A_T = self.mem_layers[-1].C_T
                C_T = lasagne.init.Normal(**std)
            else:  # RNN style
                A = self.mem_layers[-1].A
                C = self.mem_layers[-1].C
                A_T = self.mem_layers[-1].A_T
                C_T = self.mem_layers[-1].C_T
            self.mem_layers += [MemoryNetworkLayer((l_context_in,
                                                    self.mem_layers[-1],
                                                    l_context_pe_in),
                                                   vocab, embedding_size,
                                                   A=A, A_T=A_T, C=C, C_T=C_T,
                                                   nonlinearity=nonlinearity)]

        if self.adj_weight_tying:
            l_pred = TransposedDenseLayer(self.mem_layers[-1],
                                          self.num_classes,
                                          W=self.mem_layers[-1].C,
                                          b=None,
                                          nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_pred = lasagne.layers.DenseLayer(self.mem_layers[-1],
                                               self.num_classes,
                                               W=lasagne.init.Normal(**std),
                                               b=None,
                                               nonlinearity=lasagne.nonlinearities.softmax)

        parms = {l_context_in: cc, l_question_in: qq,
                 l_context_pe_in: c_pe, l_question_pe_in: q_pe}
        probas = lasagne.layers.helper.get_output(l_pred, parms)
        probas = T.clip(probas, 1e-7, 1.0-1e-7)

        pred = T.argmax(probas, axis=1)

        cost = T.nnet.binary_crossentropy(probas, y).sum()

        params = lasagne.layers.helper.get_all_params(l_pred, trainable=True)
        print 'params:', params
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(grads,
                                                             self.max_norm)
        updates = lasagne.updates.sgd(scaled_grads, params,
                                      learning_rate=self.lr)

        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared,
            c_pe: self.c_pe_shared,
            q_pe: self.q_pe_shared
        }

        self.train_model = theano.function([], cost, givens=givens,
                                           updates=updates)
        self.compute_pred = theano.function([], pred, givens=givens,
                                            on_unused_input='ignore')

        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(embedding_size,
                                 dtype=theano.config.floatX)
        updates = [(x, T.set_subtensor(x[0, :], zero_vec_tensor)) for x in [B]]
        self.set_zero = theano.function([zero_vec_tensor],
                                        updates=updates)

        self.nonlinearity = nonlinearity
        self.network = l_pred

    def reset_zero(self):
        self.set_zero(self.zero_vec)
        for l in self.mem_layers:
            l.reset_zero()

    def predict(self, dataset, index):
        self.set_shared_variables(dataset, index)
        return self.compute_pred()

    def compute_f1(self, dataset):
        n_batches = len(dataset['Y']) // self.batch_size
        _y_pred = [self.predict(dataset, i) for i in xrange(n_batches)]
        y_pred = np.concatenate(_y_pred).astype(np.int32) - 1
        y_true = [self.vocab.index(y) for y in dataset['Y'][:len(y_pred)]]
        print metrics.confusion_matrix(y_true, y_pred)
        print metrics.classification_report(y_true, y_pred)
        errors = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                errors.append((i, self.lb.classes_[p]))
        met = metrics.f1_score(y_true, y_pred,
                               average='weighted', pos_label=None)
        return met, errors

    def train(self, n_epochs=100, shuffle_batch=False):
        epoch = 0
        n_train_batches = len(self.data['train']['Y']) // self.batch_size
        self.lr = self.init_lr
        prev_train_f1 = None

        while (epoch < n_epochs):
            epoch += 1

            if epoch % 25 == 0:
                self.lr /= 2.0

            indices = range(n_train_batches)
            if shuffle_batch:
                self.shuffle_sync(self.data['train'])

            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'], minibatch_index)
                total_cost += self.train_model()
                self.reset_zero()
            end_time = time.time()
            print '\n' * 3, '*' * 80
            print 'epoch:', epoch
            print ' cost:', (total_cost / len(indices))
            print ' took: %d(s)' % (end_time - start_time)

            print 'TRAIN', '=' * 40
            train_f1, train_errors = self.compute_f1(self.data['train'])
            print 'TRAIN_ERROR:', (1-train_f1)*100
            for i, pred in train_errors[:10]:
                print 'context: ', self.to_words(self.data['train']['C'][i])
                print 'question: ', self.to_words([self.data['train']['Q'][i]])
                print 'correct answer: ', self.data['train']['Y'][i]
                print 'predicted answer: ', pred
                print '---' * 20

            if prev_train_f1 is not None and train_f1 < prev_train_f1 and self.nonlinearity is None:
                prev_weights = lasagne.layers.helper.get_all_param_values(self.network)
                self.build_network(nonlinearity=lasagne.nonlinearities.softmax)
                lasagne.layers.helper.set_all_param_values(self.network, prev_weights)
            else:
                print 'TEST', '=' * 40
                valid_f1, valid_errors = self.compute_f1(self.data['valid'])
                print '*** TEST_ERROR:', (1-valid_f1)*100

            prev_train_f1 = train_f1
        mparms = lasagne.layers.helper.get_all_param_values(self.network)
        self.ml.save_model('rnn', mparms, high=True)

    def to_words(self, indices):
        sents = []
        for idx in indices:
            words = ' '.join([self.idx2word[idx] for idx in self.S[idx] if idx > 0])
            sents.append(words)
        return ' '.join(sents)

    def shuffle_sync(self, dataset):
        p = np.random.permutation(len(dataset['Y']))
        for k in ['C', 'Q', 'Y']:
            dataset[k] = dataset[k][p]

    def set_shared_variables(self, dataset, index):
        _shape = (self.batch_size, self.maxseq)
        c = np.zeros(_shape, dtype=np.int32)

        _shape = (self.batch_size, )
        q = np.zeros(_shape, dtype=np.int32)

        _shape = (self.batch_size, self.num_classes)
        y = np.zeros(_shape, dtype=np.int32)

        _shape = (self.batch_size, self.maxseq,
                  self.maxsent, self.embedding_size)
        c_pe = np.zeros(_shape, dtype=theano.config.floatX)

        _shape = (self.batch_size, 1, self.maxsent, self.embedding_size)
        q_pe = np.zeros(_shape, dtype=theano.config.floatX)

        indices = range(index*self.batch_size, (index+1)*self.batch_size)
        for i, row in enumerate(dataset['C'][indices]):
            row = row[:self.maxseq]
            c[i, :len(row)] = row

        q[:len(indices)] = dataset['Q'][indices]

        for key, mask in [('C', c_pe), ('Q', q_pe)]:
            for i, row in enumerate(dataset[key][indices]):
                sentences = self.S[row].reshape((-1, self.maxsent))
                for ii, word_idxs in enumerate(sentences):
                    J = np.count_nonzero(word_idxs)
                    for j in np.arange(J):
                        mask[i, ii, j, :] = (1 - (j+1)/J) - ((np.arange(self.embedding_size)+1)/self.embedding_size)*(1 - 2*(j+1)/J)

        y[:len(indices), 1:self.num_classes] = self.lb.transform(dataset['Y'][indices])

        self.c_shared.set_value(c)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)
        self.c_pe_shared.set_value(c_pe)
        self.q_pe_shared.set_value(q_pe)


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


class Pred:
    def __init__(self):
        self.ml = ml.MLTools()
        self.network = None

    def pred(self, data, network, fmodel='rnn.pkl'):
        prev_weights = self.ml.read_model(fmodel)
        lasagne.layers.helper.set_all_param_values(network, prev_weights)

def to_words(idx2word, S, indices):
    sents = []
    for idx in indices:
        words = ' '.join([idx2word[idx2] for idx2 in S[idx]])
        sents.append(words)
    return ' '.join(sents)


def main():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)
    parser.add_argument('--task', type=int, default=1, help='Task#')
    parser.add_argument('--trainf', type=str, default='', help='Train file')
    parser.add_argument('--validf', type=str, default='', help='Test file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--embedding_size', type=int, default=20, help='Embedding size')
    parser.add_argument('--max_norm', type=float, default=40.0, help='Max norm')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--num_hops', type=int, default=3, help='Num hops')
    parser.add_argument('--adj_weight_tying', type='bool', default=True, help='Whether to use adjacent weight tying')
    parser.add_argument('--linear_start', type='bool', default=False, help='Whether to start with linear activations')
    parser.add_argument('--shuffle_batch', type='bool', default=True, help='Whether to shuffle minibatches')
    parser.add_argument('--n_epochs', type=int, default=100, help='Num epochs')
    args = parser.parse_args()
    print '*' * 80
    print 'args:', args
    print '*' * 80

    if args.trainf == '' or args.validf == '':
        args.trainf = glob.glob('data/en/qa%d_*train.txt' % args.task)[0]
        args.validf = glob.glob('data/en/qa%d_*test.txt' % args.task)[0]

    model = Model(**args.__dict__)
    pred = Pred()
    '''
    count = []
    word2idx = {}
    maxseq = 0
    maxsent = 0
    pdata, maxseq, maxsent = read_data_qa('tmp.txt', count, word2idx,
                                          maxseq, maxsent)
    vocab = word2idx.keys()
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    data = process_dataset(pdata, word2idx, maxsent)
    for i in range(0, len(data['Y'])):
        print(data['C'][i])
        print to_words(idx2word, data['S'], data['C'][i])
        print to_words(idx2word, data['S'], [data['Q'][i]])
        print data['Y'][i]
    '''
    # pred.pred(data, model.network)
    model.train(n_epochs=args.n_epochs, shuffle_batch=args.shuffle_batch)

if __name__ == '__main__':
    main()
