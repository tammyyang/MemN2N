from __future__ import division
import logging
import lasagne
import nltk
import numpy as np
import theano
import theano.tensor as T
import time
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from simdat.core import ml
from data import read_data_qa
from data import process_dataset
from data import print_words
from theano.printing import Print as pp


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

    def __init__(self, incomings, vocab, emb_size, A, A_T, C, C_T,
                 nonlinearity=lasagne.nonlinearities.softmax, **kwargs):
        super(MemoryNetworkLayer, self).__init__(incomings, **kwargs)
        if len(incomings) != 3:
            raise NotImplementedError

        batch_size, maxseq, maxsent = self.input_shapes[0]

        l_context_in = lasagne.layers.InputLayer(
                            shape=(batch_size, maxseq, maxsent))
        l_B_embedding = lasagne.layers.InputLayer(
                            shape=(batch_size, emb_size))
        l_context_pe_in = lasagne.layers.InputLayer(
                            shape=(batch_size, maxseq, maxsent, emb_size))
        l_context_in = lasagne.layers.ReshapeLayer(
                            l_context_in,
                            shape=(batch_size * maxseq * maxsent, ))

        l_A_embedding = lasagne.layers.EmbeddingLayer(
                            l_context_in, len(vocab)+1, emb_size, W=A)
        self.A = l_A_embedding.W
        l_A_embedding = lasagne.layers.ReshapeLayer(
                            l_A_embedding,
                            shape=(batch_size, maxseq, maxsent, emb_size))
        l_A_embedding = lasagne.layers.ElemwiseMergeLayer(
                            (l_A_embedding, l_context_pe_in),
                            merge_function=T.mul)
        l_A_embedding = SumLayer(l_A_embedding, axis=2)
        l_A_embedding = TemporalEncodingLayer(l_A_embedding, T=A_T)
        self.A_T = l_A_embedding.T

        l_C_embedding = lasagne.layers.EmbeddingLayer(
                            l_context_in, len(vocab)+1, emb_size, W=C)
        self.C = l_C_embedding.W
        l_C_embedding = lasagne.layers.ReshapeLayer(
                            l_C_embedding,
                            shape=(batch_size, maxseq, maxsent, emb_size))
        l_C_embedding = lasagne.layers.ElemwiseMergeLayer(
                            (l_C_embedding, l_context_pe_in),
                            merge_function=T.mul)
        l_C_embedding = SumLayer(l_C_embedding, axis=2)
        l_C_embedding = TemporalEncodingLayer(l_C_embedding, T=C_T)
        self.C_T = l_C_embedding.T

        l_prob = InnerProductLayer(
                    (l_A_embedding, l_B_embedding),
                    nonlinearity=nonlinearity)
        l_weighted_output = BatchedDotLayer((l_prob, l_C_embedding))

        l_sum = lasagne.layers.ElemwiseSumLayer(
                    (l_weighted_output, l_B_embedding))

        self.l_context_in = l_context_in
        self.l_B_embedding = l_B_embedding
        self.l_context_pe_in = l_context_pe_in
        self.network = l_sum

        params = lasagne.layers.helper.get_all_params(
                    self.network, trainable=True)
        values = lasagne.layers.helper.get_all_param_values(
                    self.network, trainable=True)
        for p, v in zip(params, values):
            self.add_param(p, v.shape, name=p.name)

        zero_tensor = T.vector()
        self.zero_vec = np.zeros(emb_size,
                                 dtype=theano.config.floatX)
        updates = [(x, T.set_subtensor(x[0, :], zero_tensor))
                   for x in [self.A, self.C]]
        self.set_zero = theano.function(
                        [zero_tensor],
                        updates=updates)

    def get_output_shape_for(self, input_shapes):
        return lasagne.layers.helper.get_output_shape(self.network)

    def get_output_for(self, inputs, **kwargs):
        inputs = {self.l_context_in: inputs[0],
                  self.l_B_embedding: inputs[1],
                  self.l_context_pe_in: inputs[2]}
        output = lasagne.layers.helper.get_output(self.network, inputs)
        return output

    def reset_zero(self):
        self.set_zero(self.zero_vec)


class RNNArgs(ml.Args):
    def _add_args(self):
        """Called by __init__ of Args class"""
        self.trainf = 'data/en/qa1_single-supporting-fact_train.txt'
        self.validf = 'data/en/qa1_single-supporting-fact_test.txt'
        self.batch_size = 32
        self.emb_size = 20
        self.max_norm = 40
        self.lr = 0.01
        self.linear_start = True
        self.nepochs = 100
        self.shuffle_batch = False
        self.adj_w_tying = True
        self.nhops = 3


class Model:
    def __init__(self):
        self.ml = ml.MLTools()
        self.args = RNNArgs(pfs=['rnn.json'])
        self.count = []
        self.S = []
        self.word2idx = {}
        self.idx2word = {}
        self.data = {'train': {}, 'valid': {}}
        self.maxseq = 0
        self.maxsent = 0
        self.num_classes = 0
        self.vocab = None
        self.lb = None
        self.init_lr = self.args.lr
        self.lr = self.args.lr
        self.nonlinearity = None

    def init_train(self):
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
        self.vocab = self.word2idx.keys()

        print('[N2N] batch_size: %i' % self.args.batch_size)
        print('[N2N] maxseq: %i' % maxseq)
        print('[N2N] maxsent: %i' % maxsent)
        print('[N2N] sentences:')
        print(self.S.shape)
        print('[N2N] vocab: %i' % len(self.vocab))
        print(self.vocab)
        for d in ['train', 'valid']:
            print d,
            for k in ['C', 'Q', 'Y']:
                print k, self.data[d][k].shape,
            print ''

        lb = LabelBinarizer()
        lb.fit(list(self.vocab))
        self.vocab = lb.classes_.tolist()

        self.maxseq = maxseq
        self.maxsent = maxsent
        self.num_classes = len(self.vocab) + 1
        self.lb = lb
        if not self.args.linear_start:
            self.nonlinearity = lasagne.nonlinearities.softmax

        self.build_network(self.nonlinearity)

    def init_data(self, fdata, maxseq=0, maxsent=0):

        data, maxseq, maxsent = read_data_qa(fdata, self.count,
                                             self.word2idx,
                                             maxseq, maxsent)
        processed = process_dataset(data, self.word2idx, maxsent,
                                    offset=len(self.S))
        self.S.extend(processed['S'])
        return maxseq, maxsent, processed

    def build_network(self, nonlinearity):
        batch_size = self.args.batch_size
        maxseq, maxsent = self.maxseq, self.maxsent
        emb_size, vocab = self.args.emb_size, self.vocab

        c = T.imatrix()
        q = T.ivector()
        y = T.imatrix()
        c_pe = T.tensor4()
        q_pe = T.tensor4()

        self.c_shared = theano.shared(
                            np.zeros((batch_size, maxseq),
                                     dtype=np.int32),
                            borrow=True)

        self.q_shared = theano.shared(
                            np.zeros((batch_size, ),
                                     dtype=np.int32),
                            borrow=True)

        self.a_shared = theano.shared(
                            np.zeros((batch_size, self.num_classes),
                                     dtype=np.int32),
                            borrow=True)

        self.c_pe_shared = theano.shared(
                            np.zeros((batch_size, maxseq, maxsent, emb_size),
                                     dtype=theano.config.floatX),
                            borrow=True)

        self.q_pe_shared = theano.shared(
                            np.zeros((batch_size, 1, maxsent, emb_size),
                                     dtype=theano.config.floatX),
                            borrow=True)

        S_shared = theano.shared(self.S, borrow=True)
        cc = S_shared[c.flatten()].reshape((batch_size, maxseq, maxsent))
        qq = S_shared[q.flatten()].reshape((batch_size, maxsent))
        l_context_in = lasagne.layers.InputLayer(
                            shape=(batch_size, maxseq, maxsent))
        l_question_in = lasagne.layers.InputLayer(
                            shape=(batch_size, maxsent))
        l_context_pe_in = lasagne.layers.InputLayer(
                            shape=(batch_size, maxseq, maxsent, emb_size))
        l_question_pe_in = lasagne.layers.InputLayer(
                            shape=(batch_size, 1, maxsent, emb_size))

        std = {'std': 0.1}
        A = lasagne.init.Normal(**std).sample((len(vocab)+1, emb_size))
        C = lasagne.init.Normal(**std)
        A_T = lasagne.init.Normal(**std)
        C_T = lasagne.init.Normal(**std)
        W = A if self.args.adj_w_tying else lasagne.init.Normal(**std)

        l_question_in = lasagne.layers.ReshapeLayer(
                            l_question_in, shape=(batch_size * maxsent, ))
        l_B_embedding = lasagne.layers.EmbeddingLayer(
                            l_question_in, len(vocab)+1,
                            emb_size, W=W)
        B = l_B_embedding.W
        l_B_embedding = lasagne.layers.ReshapeLayer(
                            l_B_embedding,
                            shape=(batch_size, 1, maxsent, emb_size))
        l_B_embedding = lasagne.layers.ElemwiseMergeLayer(
                            (l_B_embedding, l_question_pe_in),
                            merge_function=T.mul)
        _shape = (batch_size, maxsent, emb_size)
        l_B_embedding = lasagne.layers.ReshapeLayer(
                            l_B_embedding, shape=_shape)
        l_B_embedding = SumLayer(l_B_embedding, axis=1)

        self.mem_layers = [MemoryNetworkLayer(
                            (l_context_in, l_B_embedding, l_context_pe_in),
                            vocab, emb_size,
                            A=A, A_T=A_T, C=C, C_T=C_T,
                            nonlinearity=nonlinearity)]

        for _ in range(1, self.args.nhops):
            if self.args.adj_w_tying:
                A = self.mem_layers[-1].C
                C = lasagne.init.Normal(**std)
                A_T = self.mem_layers[-1].C_T
                C_T = lasagne.init.Normal(**std)
            else:  # RNN style
                A = self.mem_layers[-1].A
                C = self.mem_layers[-1].C
                A_T = self.mem_layers[-1].A_T
                C_T = self.mem_layers[-1].C_T
            self.mem_layers.append(
                MemoryNetworkLayer(
                    (l_context_in, self.mem_layers[-1], l_context_pe_in),
                    vocab, emb_size,
                    A=A, A_T=A_T,
                    C=C, C_T=C_T,
                    nonlinearity=nonlinearity))

        if self.args.adj_w_tying:
            l_pred = TransposedDenseLayer(
                        self.mem_layers[-1],
                        self.num_classes,
                        W=self.mem_layers[-1].C,
                        b=None,
                        nonlinearity=lasagne.nonlinearities.softmax)
        else:
            l_pred = lasagne.layers.DenseLayer(
                        self.mem_layers[-1],
                        self.num_classes,
                        W=lasagne.init.Normal(**std),
                        b=None,
                        nonlinearity=lasagne.nonlinearities.softmax)

        parms = {l_context_in: cc,
                 l_question_in: qq,
                 l_context_pe_in: c_pe,
                 l_question_pe_in: q_pe}
        probas = lasagne.layers.helper.get_output(l_pred, parms)
        probas = T.clip(probas, 1e-7, 1.0-1e-7)

        pred = T.argmax(probas, axis=1)

        cost = T.nnet.binary_crossentropy(probas, y).sum()

        params = lasagne.layers.helper.get_all_params(
                        l_pred, trainable=True)
        grads = T.grad(cost, params)
        scaled_grads = lasagne.updates.total_norm_constraint(
                        grads, self.args.max_norm)
        updates = lasagne.updates.sgd(
                        scaled_grads, params,
                        learning_rate=self.lr)

        givens = {
            c: self.c_shared,
            q: self.q_shared,
            y: self.a_shared,
            c_pe: self.c_pe_shared,
            q_pe: self.q_pe_shared
        }

        self.train_model = theano.function(
                            [], cost, givens=givens,
                            updates=updates)
        self.compute_pred = theano.function(
                            [], pred, givens=givens,
                            on_unused_input='ignore')

        zero_tensor = T.vector()
        self.zero_vec = np.zeros(emb_size,
                                 dtype=theano.config.floatX)
        updates = [(x, T.set_subtensor(x[0, :], zero_tensor)) for x in [B]]
        self.set_zero = theano.function([zero_tensor],
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
        n_batches = len(dataset['Y']) // self.args.batch_size
        _y_pred = [self.predict(dataset, i) for i in xrange(n_batches)]
        y_pred = np.concatenate(_y_pred).astype(np.int32) - 1
        y_true = [self.vocab.index(y) for y in dataset['Y'][:len(y_pred)]]
        logging.debug([self.vocab[i] for i in y_pred if i not in y_true])
        logging.debug(metrics.confusion_matrix(y_true, y_pred))
        logging.debug(metrics.classification_report(y_true, y_pred))
        errors = []
        for i, (t, p) in enumerate(zip(y_true, y_pred)):
            if t != p:
                errors.append((i, self.lb.classes_[p]))
        met = metrics.f1_score(y_true, y_pred,
                               average='weighted', pos_label=None)
        return met, errors

    def train(self):
        self.init_train()
        epoch = 0
        n_train_batches = len(self.data['train']['Y']) // self.args.batch_size
        self.lr = self.init_lr
        prev_train_f1 = None

        while (epoch < self.args.nepochs):
            epoch += 1

            if epoch % 25 == 0:
                self.lr /= 2.0

            indices = range(n_train_batches)
            if self.args.shuffle_batch:
                self.shuffle_sync(self.data['train'])

            total_cost = 0
            start_time = time.time()
            for minibatch_index in indices:
                self.set_shared_variables(self.data['train'],
                                          minibatch_index)
                total_cost += self.train_model()
                self.reset_zero()
            end_time = time.time()
            print('[N2N] %s' % ('='*40))
            print('[N2N] epoch: %i' % epoch)
            print('[N2N]   cost: %.2f' % (total_cost / len(indices)))
            print('[N2N]   took: %d(s)' % (end_time - start_time))

            print('[N2N] --- TRAIN ---')
            train_f1, train_errors = self.compute_f1(self.data['train'])
            print('[N2N]   error: %.2f' % ((1-train_f1)*100))
            for i, pred in train_errors[:10]:
                logging.debug('Context:')
                logging.debug(' %s' % print_words(self.idx2word,
                                                  self.S,
                                                  self.data['train']['C'][i]))
                logging.debug('Question:')
                logging.debug(' %s' % print_words(self.idx2word,
                                                  self.S,
                                                  [self.data['train']['Q'][i]]))
                logging.debug('Correct answer:')
                logging.debug(' %s' % self.data['train']['Y'][i])
                logging.debug('Predicted answer: ')
                logging.debug(' %s' % pred)
                logging.debug('%s' % ('-'*60))

            train_epoch = True
            if prev_train_f1 is None:
                train_epoch = False
            if train_f1 > prev_train_f1:
                train_epoch = False
            if self.nonlinearity is not None:
                train_epoch = False

            if train_epoch:
                prev_weights = lasagne.layers.helper.get_all_param_values(self.network)
                self.build_network(nonlinearity=lasagne.nonlinearities.softmax)
                lasagne.layers.helper.set_all_param_values(self.network,
                                                           prev_weights)
            else:
                print('[N2N] --- TEST ---')
                valid_f1, valid_errors = self.compute_f1(self.data['valid'])
                print('[N2N]   error: %.2f' % ((1-valid_f1)*100))

            prev_train_f1 = train_f1
        mparms = lasagne.layers.helper.get_all_param_values(self.network)
        self.ml.save_model('rnn', mparms, high=True)

    def shuffle_sync(self, dataset):
        p = np.random.permutation(len(dataset['Y']))
        for k in ['C', 'Q', 'Y']:
            dataset[k] = dataset[k][p]

    def set_shared_variables(self, dataset, index):
        c = np.zeros((self.args.batch_size, self.maxseq),
                     dtype=np.int32)

        q = np.zeros((self.args.batch_size, ),
                     dtype=np.int32)

        y = np.zeros((self.args.batch_size, self.num_classes),
                     dtype=np.int32)

        c_pe = np.zeros((self.args.batch_size,
                         self.maxseq,
                         self.maxsent,
                         self.args.emb_size),
                        dtype=theano.config.floatX)

        q_pe = np.zeros((self.args.batch_size,
                         1,
                         self.maxsent,
                         self.args.emb_size),
                        dtype=theano.config.floatX)

        indices = range(index*self.args.batch_size,
                        (index+1)*self.args.batch_size)
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
                        mask[i, ii, j, :] = self.cal_mi(j, J)

        y[:len(indices), 1:self.num_classes] = \
            self.lb.transform(dataset['Y'][indices])

        self.c_shared.set_value(c)
        self.q_shared.set_value(q)
        self.a_shared.set_value(y)
        self.c_pe_shared.set_value(c_pe)
        self.q_pe_shared.set_value(q_pe)

    def cal_mi(self, j, J):
        d = self.args.emb_size  # page 4, 5
        return (1-(j+1)/J) - ((np.arange(d)+1)/d)*(1-2*(j+1)/J)


class Pred:
    def __init__(self):
        self.ml = ml.MLTools()
        self.network = None

    def pred(self, data, network, fmodel='rnn.pkl'):
        prev_weights = self.ml.read_model(fmodel)
        lasagne.layers.helper.set_all_param_values(network, prev_weights)
