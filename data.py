import os
import nltk
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict


def print_words(idx2word, S, indices):
    sents = []
    for idx in indices:
        words = ' '.join([idx2word[idx2] for idx2 in S[idx]])
        sents.append(words)
    return ' '.join(sents)


def get_statement(line):
    line = line.replace('\n', '')
    bk = line.find(' ')
    lid = int(line[:bk+1]) - 1
    segs = line.split('?')
    try:
        ans = segs[1].split('\t')
        ans = [s.replace(' ', '') for s in ans]
        return segs[0][bk+1:], [ans[-1], ans[-2]], lid
    except IndexError:
        return segs[0][bk+1:], None, lid


def process_dataset(data, word2idx, maxsent, offset=0):
    S, Y, C, Q = [], [], [], []
    for stat in data[-1]:
        indices = map(lambda x: word2idx[x], stat)
        indices += [0] * (maxsent - len(stat))
        S.append(indices)
    for i in data.keys():
        if i < 0:
            continue
        C.append([idx + offset for idx in data[i][3]])
        Q.append(i + offset)
        Y.append(data[i][1][1])
    return {'S': S,
            'Y': np.array(Y), 'C': np.array(C),
            'Q': np.array(Q, dtype=np.int32)}


def read_data_qa(fname, count, word2idx,
                 max_seqlen, max_sentlen):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    stats = []
    stats_idx = []
    all_stats = []
    data = OrderedDict()
    counter = 0
    start_idx = 0
    for i in range(0, len(lines)):
        line = lines[i]
        stat, ans, lid = get_statement(line)
        _stats = nltk.word_tokenize(stat)
        max_sentlen = max(len(_stats), max_sentlen)
        words.extend(_stats)
        if lid == 0:
            max_seqlen = max(counter, max_seqlen)
            counter = 0
            start_idx = i
            stats_idx = []
        all_stats.append(_stats)
        if ans is None:
            counter += 1
            stats_idx.append(i)
        else:
            data[i] = [deepcopy(stat), ans,
                       list(reversed(all_stats[start_idx:i])),
                       list(reversed(stats_idx))]

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data[-1] = all_stats
    return data, max_seqlen, max_sentlen
