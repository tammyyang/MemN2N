import os
import nltk
import numpy as np
from copy import deepcopy
from collections import Counter
from collections import OrderedDict

def get_statement(line):
    line = line.replace('\n', '')
    bk = line.find(' ')
    lid = int(line[:bk+1])
    segs = line.split('?')
    try:
        return segs[0][bk+1:], segs[1], lid
    except IndexError:
        return segs[0][bk+1:], None, lid

def process_dataset(data, word2idx, maxsent):
    S, Y = [], []
    for stat in data[-1]:
        indices = map(lambda x: word2idx[x], stat)
        indices += [0] * (maxsent - len(stat))
        S.append(indices)
    S = np.array(S, dtype=np.int32)
    C = [data[i][2] for i in data.keys() if i >= 0]
    Q = [i for i in data.keys() if i >= 0]

    '''
    for key, value in data.items():
        if key == -1:
            continue
    for i, qline in enumerate(data):
        if line['type'] == 'q':
            id = line['id']-1
            indices = [offset+idx+1 for idx in range(i-id, i) if lines[idx]['type'] ==
's'][::-1][:50]
            line['refs'] = [indices.index(offset+i+1-id+ref) for ref in line['refs']]
            C.append(indices)
            Q.append(offset+i+1)
            Y.append(line['answer'])
    # return np.array(S, dtype=np.int32), np.array(C), np.array(Q, dtype=np.int32),
#np.array(Y)
    '''

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
        if lid == 1:
            max_seqlen = max(counter, max_seqlen)
            counter == 0
            start_idx = i
            stats_idx = []
        all_stats.append(_stats)
        if ans is None:
            counter += 1
            stats_idx.append(i + 1)
        else:
            print(ans)
            data[i + 1] = [deepcopy(stat),
                           list(reversed(all_stats[start_idx:i])),
                           list(reversed(stats_idx))]

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    for word, _ in count:
        if not word2idx.has_key(word):
            word2idx[word] = len(word2idx)

    # print(count)
    # print(word2idx)
    data[-1] = all_stats
    return data, max_seqlen, max_sentlen
