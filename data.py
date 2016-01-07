import os
import nltk
from collections import Counter

def get_statement(line):
    bk = line.find(' ')
    lid = int(line[:bk+1])
    segs = line.split('?')
    try:
        return segs[0][bk+1:], segs[1], lid
    except IndexError:
        return segs[0][bk+1:], None, lid

def read_data_qa(fname, count, word2idx, max_seqlen):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    stats = []
    counter = 0
    for line in lines:
        stat, ans, lid = get_statement(line)
        _stats = nltk.word_tokenize(stat)
        words.extend(_stats)
        stats.append(_stats)
        if lid == 1:
            max_seqlen = max(counter, max_seqlen)
            print(counter, max_seqlen)
            counter == 0
        if ans is None:
            counter += 1

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    for word, _ in count:
        if not word2idx.has_key(word):
            word2idx[word] = len(word2idx)

    data = list()
    for stat in stats:
        for word in stat:
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data
