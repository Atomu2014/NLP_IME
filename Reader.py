import cPickle as pk
import os
import re
import time

import numpy as np
import scipy.sparse as spr

letters = {'\'': 1, '<s>': 2, 'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7, 'f': 8, 'g': 9, 'h': 10, 'i': 11, 'j': 12,
           'k': 13, 'l': 14, 'm': 15, 'n': 16, 'o': 17, 'p': 18, 'q': 19, 'r': 20, 's': 21, 't': 22, 'u': 23, 'v': 24,
           'w': 25, 'x': 26, 'y': 27, 'z': 28}
letters_num = len(letters) + 1
word_length = 30
op_costs = {'s': 0, 'a': 1, 'd': 2, 't': 3}


def get_kb_map():
    raw_board = ['1234567890-=',
                 'qwertyuiop[]\\',
                 'asdfghjkl;\'',
                 'zxcvbnm,./']

    kb_map = {}
    for r in range(4):
        for i in range(len(raw_board[r])):
            nbs = ''
            if i > 0:
                nbs += raw_board[r][i - 1]
            if i + 1 < len(raw_board[r]):
                nbs += raw_board[r][i + 1]
            if r > 0:
                if i < len(raw_board[r - 1]):
                    nbs += raw_board[r - 1][i]
                if i + 1 < len(raw_board[r - 1]):
                    nbs += raw_board[r - 1][i + 1]
            if r < 3:
                if i < len(raw_board[r + 1]):
                    nbs += raw_board[r + 1][i]
                if 0 <= i - 1 < len(raw_board[r + 1]):
                    nbs += raw_board[r + 1][i - 1]
            kb_map[raw_board[r][i]] = nbs

    return kb_map


def split_train(train_set, alpha=0.1):
    import random
    with open(train_set, 'r') as fin:
        with open(train_set + '.part1', 'w') as out1:
            with open(train_set + '.part2', 'w') as out2:
                for line in fin:
                    line = line.strip()
                    if random.random() <= alpha:
                        out1.write(line + '\n')
                    else:
                        out2.write(line + '\n')


def split_last_words(line, num=3, cs=False):
    if not cs:
        line = line.strip().lower()
    else:
        line = line.strip()
    lastindex = len(line)
    res = []
    for i in range(num):
        index = line.rfind(' ', 0, lastindex)
        if index != -1:
            word = line[index + 1:lastindex]
        else:
            word = '<s>'
        res.append(word)
        lastindex = index
    return res


def is_ascii(s):
    return all(ord(c) < 128 for c in s)


def make_sens():
    with open('raw/corpus', 'r') as fin:
        with open('raw/corpus.sens', 'w') as fout:
            for line in fin:
                if is_ascii(line):
                    for sen in re.findall("[a-z ']+", line.lower().strip()):
                        sen = sen.strip()
                        if len(sen) > 0:
                            fout.write(sen + '\n')


def make_word_vector(word):
    if word == '<s>':
        return [letters[word]]

    vect = []
    for c in word[:word_length]:
        if c in letters:
            vect.append(letters[c])
        else:
            vect.append(0)
    return vect


def clean_sens(dict_in, corpus_out):
    with open('raw/corpus.sens', 'r') as corpus_in:
        with open(corpus_out, 'w') as corpus_out:
            dict = set(np.loadtxt(dict_in, dtype=str))

            for line in corpus_in:
                line = line.strip()
                flag = True
                for word in line.split():
                    if word not in dict:
                        flag = False
                        break
                if flag:
                    corpus_out.write(line + '\n')


def count_word_length():
    # 42 23 23 -> 30
    with open('raw/train', 'r') as fin:
        with open('raw/log', 'w') as log_out:
            len1 = 0
            len2 = 0
            len3 = 0
            dict = []
            for line in fin:
                label, last, last2 = split_last_words(line)

                len1 = max(len1, len(last2))
                len2 = max(len2, len(last))
                len3 = max(len3, len(label))

                dict.append(last2)

            dict = sorted(set(dict), key=lambda x: -len(x))
            for word in dict:
                log_out.write(word + '\n')

        print len1, len2, len3


def count_word():
    with open('raw/corpus.sens', 'r') as fin:
        dict = {}
        for line in fin:
            for word in re.findall("[a-z']+", line.strip()):
                if word in dict:
                    dict[word] += 1
                else:
                    dict[word] = 1

        print 'dict size', len(dict)

        with open('raw/corpus.words', 'w') as fout:
            for key, value in sorted(dict.items(), key=lambda (term, freq): -freq):
                fout.write(key + '\t' + str(value) + '\n')


def count_ngram(N, fin, fout):
    print 'count', N, '-gram on', fin, 'out to', fout
    import pynlpl.textprocessors as proc

    with open(fin, 'r') as corpus_in:
        ngrams = {}
        for line in corpus_in:
            words = proc.tokenize(line)
            for ngram in proc.Windower(words, N, '<s>', None):
                key = '\t'.join(ngram)
                if key in ngrams:
                    ngrams[key] += 1
                else:
                    ngrams[key] = 1

        with open(fout, 'w') as ngrams_out:
            for key, value in sorted(ngrams.items(), key=lambda (term, freq): (term, -freq)):
                if value > 1:
                    ngrams_out.write(key + '\t' + str(value) + '\n')


def count_ngram2(N, fin, fout):
    print 'count', N, '-gram2 on', fin, 'out to', fout, 'only top candidate reserved'
    import pynlpl.textprocessors as proc

    with open(fin, 'r') as corpus_in:
        ngrams = {}
        count_total = 0
        for line in corpus_in:
            words = proc.tokenize(line)
            for ngram in proc.Windower(words, N, '<s>', None):
                term = '\t'.join(ngram[:-1])
                cand = ngram[-1]
                if term in ngrams:
                    if cand in ngrams[term]:
                        ngrams[term][cand] += 1
                    else:
                        ngrams[term][cand] = 1
                else:
                    ngrams[term] = {cand: 1}

            count_total += 1
            if count_total % 1000000 == 0:
                print count_total

        for k in ngrams.keys():
            ngrams[k] = sorted(ngrams[k].items(), key=lambda (_, freq): (-freq))[0][0]

        with open(fout, 'w') as ngrams_out:
            for key, value in sorted(ngrams.items(), key=lambda (term, cand): (term, cand)):
                ngrams_out.write(key + '\t' + value + '\n')


def count_ngram3(train_set, max_len=6, cs=False):
    with open(train_set, 'r') as train_in:
        ngrams_dict = [{} for i in range(max_len)]
        for line in train_in:
            words = line.strip().lower().split('<s>')[-1].split(' ')
            label = split_last_words(line, 1, cs)[0]
            if words[0] != '<s>':
                words = ['<s>'] + words
            words = words[:-1]

            for i in range(min(max_len, len(words))):
                term = ' '.join(words[len(words) - 1 - i:])
                if term in ngrams_dict[i]:
                    if label in ngrams_dict[i][term]:
                        ngrams_dict[i][term][label] += 1
                    else:
                        ngrams_dict[i][term][label] = 1
                else:
                    ngrams_dict[i][term] = {label: 1}

        for i in range(len(ngrams_dict)):
            for k in ngrams_dict[i].keys():
                ngrams_dict[i][k] = sorted(ngrams_dict[i][k].items(), reverse=True, key=lambda (a, b): b)[0][0]
    fout_path = train_set + '.ngram.bin'
    if cs:
        fout_path += '.cs'
    pk.dump(ngrams_dict, open(fout_path, 'w'))


def load_dictionary(dict):
    import pandas as pd
    return pd.read_csv(dict, names=['word'], dtype={'word': str})


def load_additional_dict(corpus, train_set):
    print 'load additional dictionary', corpus, train_set

    add_dict_path = train_set + '.adddict'
    if os.path.isfile(add_dict_path):
        return pk.load(open(add_dict_path, 'r'))

    from SymSpell import SymSpell
    sp = SymSpell(corpus, 1)

    with open(train_set, 'r') as fin:
        add_dict = {}
        cnt = sp.unique_word_count + 1
        for line in fin:
            label, last = split_last_words(line.strip().lower(), 2)
            if sp.get_index(last) == 0 and last not in add_dict:
                add_dict[last] = cnt
                cnt += 1

    pk.dump(add_dict, open(train_set + '.adddict', 'w'))
    return add_dict


def load_ngrams(bigram_path):
    print 'load n-grams', bigram_path, 'form: {dict:[list]}'

    ngrams_bin_path = bigram_path + '.bin'
    if os.path.isfile(ngrams_bin_path):
        return pk.load(open(ngrams_bin_path, 'r'))

    ngrams = {}
    with open(bigram_path, 'r') as ngram_in:
        for line in ngram_in:
            arr = line.strip().split()
            term = '\t'.join(arr[:-2])
            if term in ngrams:
                ngrams[term].append((arr[-2], int(arr[-1])))
            else:
                ngrams[term] = [(arr[-2], int(arr[-1]))]

    for key in ngrams.keys():
        ngrams[key].sort(reverse=True, key=lambda (a, b): b)

    pk.dump(ngrams, open(ngrams_bin_path, 'w'))
    return ngrams


def load_ngrams2(ngram_path):
    print 'load ngrams2', ngram_path, "form: {dict:{dict ('#total#: total_num')}}"

    ngram_bin_path = ngram_path + '.bin2'
    if os.path.isfile(ngram_bin_path):
        return pk.load(open(ngram_bin_path, 'r'))

    ngrams = {}
    with open(ngram_path, 'r') as ngram_in:
        for line in ngram_in:
            arr = line.strip().split()
            term = '\t'.join(arr[:-2])
            if term in ngrams:
                ngrams[term][arr[-2]] = int(arr[-1])
            else:
                ngrams[term] = {arr[-2]: int(arr[-1])}

    for key in ngrams.keys():
        counter = 0
        for key2 in ngrams[key]:
            counter += ngrams[key][key2]
        ngrams[key]['#total#'] = counter

    pk.dump(ngrams, open(ngram_bin_path, 'w'))
    return ngrams


def load_ngram3(ngram_path):
    print 'load ngram3', ngram_path, 'form: {term: cand}'
    ngram_bin_path = ngram_path + '.bin3'
    if os.path.isfile(ngram_bin_path):
        return pk.load(open(ngram_bin_path, 'r'))

    ngrams = {}
    with open(ngram_path, 'r') as ngram_in:
        for line in ngram_in:
            arr = line.strip().split()
            term = '\t'.join(arr[:-1])
            ngrams[term] = arr[-1]

    pk.dump(ngrams, open(ngram_bin_path, 'w'))
    return ngrams


def get_bifreq(bigrams, key, key2, norm=True, smooth=False):
    if key in bigrams:
        if key2 in bigrams[key]:
            if norm:
                return bigrams[key][key2] * 1.0 / bigrams[key]['#total#']
            else:
                return bigrams[key][key2]
    return 0


def make_feature(corpus, train_set):
    from SymSpell import SymSpell
    sp = SymSpell(corpus)

    print 'make feature', corpus, train_set
    with open(train_set, 'r') as train_in:
        with open(train_set + '.feature', 'w') as feature_out:
            bigrams = load_ngrams(corpus + '.bigram')

            count_total = 0
            count_feature = 0
            stime = time.time()

            for line in train_in:
                label, last, last2 = split_last_words(line, 3)

                count_total += 1

                if count_total % 1000 == 0:
                    etime = time.time()
                    print count_total, count_feature, etime - stime
                    stime = etime

                if sp.get_index(label) == 0:
                    continue

                sug_list = sp.get_suggestions(last, True)
                for edit, value in sug_list:
                    # label
                    # [0-len]:last2
                    # [len+1 - 2*len+1]:edit
                    # 2*len+2:is_prefix
                    # 2*len+3:edit distance
                    # 2*len+4:freq
                    # 2*len+5: bi_freq
                    count_feature += 1

                    if edit == label:
                        feature_out.write('1 ')
                    else:
                        feature_out.write('0 ')
                    feature_out.write(str(sp.get_index(last2)) + ':1 ')
                    feature_out.write(str(sp.unique_word_count + 1 + sp.get_index(edit)) + ':1 ')
                    feature_out.write(str(2 * sp.unique_word_count + 1) + ':1 ')
                    if edit.startswith(last):
                        feature_out.write(str(2 * sp.unique_word_count + 2) + ':1 ')
                    else:
                        feature_out.write(str(2 * sp.unique_word_count + 2) + ':0 ')
                    feature_out.write(str(2 * sp.unique_word_count + 3) + ':' + str(value[1]) + ' ')
                    feature_out.write(str(2 * sp.unique_word_count + 4) + ':' + str(value[0]) + ' ')
                    term = last2 + '\t' + edit
                    if term in bigrams:
                        feature_out.write(str(2 * sp.unique_word_count + 5) + ':' + str(bigrams[term]) + '\n')
                    else:
                        feature_out.write(str(2 * sp.unique_word_count + 5) + ':0\n')


def get_feature_word_vector(last2, last, edit, label, freq, bi_freq):
    # label
    # last2
    # last
    # edit
    # freq
    # bi_freq
    foo = ''
    st = 0

    vect_last2 = make_word_vector(last2)
    for i in range(len(vect_last2)):
        foo += str(st + i * letters_num + vect_last2[i]) + ':1 '
    st += word_length * letters_num
    vect_last = make_word_vector(last)
    for i in range(len(vect_last)):
        foo += str(st + i * letters_num + vect_last[i]) + ':1 '
    st += word_length * letters_num
    vect_edit = make_word_vector(edit)
    for i in range(len(vect_edit)):
        foo += str(st + i * letters_num + vect_edit[i]) + ':1 '
    st += word_length * letters_num
    foo += str(st) + ':' + str(freq) + ' '
    st += 1
    foo += str(st) + ':' + str(bi_freq) + '\n'

    return foo


def foo_2_csr(foos):
    r = 0
    data = []
    row_ind = []
    col_ind = []
    for foo in foos:
        for fea in foo.split():
            x, y = fea.split(':')
            data.append(float(y))
            col_ind.append(int(x))
            row_ind.append(r)
        r += 1
    return spr.csr_matrix((data, (row_ind, col_ind)))


def make_feature_word_vector(corpus, train_set):
    from SymSpell import SymSpell
    sp = SymSpell(corpus)

    print 'make feature word vector', corpus, train_set
    with open(train_set, 'r') as train_in:
        with open(train_set + '.vfeature2', 'w') as feature_out:
            bigrams = load_ngrams2(corpus + '.bigram')

            count_total = 0
            count_feature = long(0)
            stime = time.time()
            for line in train_in:
                label, last, last2 = split_last_words(line, 3)

                count_total += 1

                # if sp.get_index(last):
                #     continue

                if count_total % 10000 == 0:
                    etime = time.time()
                    print count_total, count_feature, etime - stime
                    stime = etime

                sug_list = sp.get_suggestions(last, True)
                for edit, _ in sug_list:
                    count_feature += 1

                    if edit == label:
                        feature_out.write('1 ')
                    else:
                        feature_out.write('0 ')
                    feature_out.write(get_feature_word_vector(last2, last, edit, label, sp.get_freq(edit),
                                                              get_bifreq(bigrams, last2, edit)))


def get_feature_edit_vector(sp, last2, last, word, ops, max_dist=1):
    foo = ''
    st = 0
    foo += str(sp.get_index(last2)) + ':1 '
    st += sp.unique_word_count + 1
    foo += str(st + sp.get_index2(last)) + ':1 '
    st += sp.unique_word_count + 1 + sp.add_dict_size
    foo += str(st + sp.get_index(word)) + ':1 '
    st += sp.unique_word_count + 1
    for i in range(max_dist):
        if i < len(ops):
            foo += str(st + op_costs[ops[i]]) + ':1 '
        else:
            foo += str(st) + ':1 '
        st += 5
    return foo


def make_feature_edit_vector(corpus, train_set, max_dist=1):
    from SymSpell import SymSpell
    sp = SymSpell(corpus, max_dist, load_additional_dict(corpus, 'raw/train'))

    print 'make feature edit vector', corpus, train_set
    with open(train_set, 'r') as train_in:
        with open(train_set + '.ellfeature' + str(max_dist), 'w') as fea_out:
            with open(train_set + '.ellfeature' + str(max_dist) + '.log', 'w') as log_out:
                count_total = 0
                stime = time.time()

                for line in train_in:
                    label, last, last2 = split_last_words(line, 3)

                    count_total += 1

                    if count_total % 10000 == 0:
                        etime = time.time()
                        print count_total, etime - stime
                        stime = etime

                    sug_list = sp.get_suggestions(last, True)
                    for edit, val in sug_list:
                        log_out.write(edit + ' ')
                        if edit == label:
                            fea_out.write('1 ')
                        else:
                            fea_out.write('0 ')
                        fea_out.write(get_feature_edit_vector(sp, last2, last, edit, val[2], max_dist) + '\n')
                    log_out.write('\n')


def get_we(model, word):
    for i in range(len(word), 0, -1):
        if word[:i] in model:
            return model[word[:i]]
    return [0] * model.vector_size


def get_feature_we(w2v_model, last2, last, pred, freq, bi_freq, ev):
    last2v = get_we(w2v_model, last2)
    lastv = get_we(w2v_model, last)
    predv = get_we(w2v_model, pred)
    if pred.startswith(last) or last.startswith(pred):
        is_prefix = 1
    else:
        is_prefix = 0

    foo = ''
    st = 0
    for i in range(len(last2v)):
        foo += str(st + i) + ':' + str(last2v[i]) + ' '
    st += len(last2v)
    for i in range(len(lastv)):
        foo += str(st + i) + ':' + str(lastv[i]) + ' '
    st += len(lastv)
    for i in range(len(predv)):
        foo += str(st + i) + ':' + str(predv[i]) + ' '
    st += len(predv)
    foo += str(st) + ':' + str(freq) + ' '
    st += 1
    foo += str(st) + ':' + str(bi_freq) + ' '
    st += 1
    for i in range(len(ev)):
        foo += str(st + i) + ':' + str(ev[i]) + ' '
    st += len(ev)
    foo += str(st) + ':' + str(is_prefix)

    return foo


def make_feature_we(corpus, ngram_path, w2v_path, train_set):
    print 'make feature word embedding'
    from SymSpell import SymSpell
    sp = SymSpell(corpus, 3)
    print 'max edit dist', 3
    from gensim.models.word2vec import Word2Vec
    w2v = Word2Vec.load(w2v_path)
    kb_map = get_kb_map()
    bigrams = load_ngrams2(ngram_path)

    with open(train_set, 'r') as train_in:
        with open(train_set + '.feature.w2v', 'w') as fea_out:
            ngrams = load_ngrams(ngram_path)

            count_total = 0
            stime = time.time()

            for line in train_in:
                label, last, last2 = split_last_words(line, 3)

                count_total += 1
                if count_total % 10000 == 0:
                    etime = time.time()
                    print count_total, etime - stime
                    stime = etime

                term = last2
                edit_list = sp.get_suggestions(last, True, True)
                sug_list = []
                freqs = {}
                evs = {}
                for ed, trace in edit_list:
                    flag = True
                    ev = [0] * len(op_costs)
                    for op in trace[2].split('<op>'):
                        if len(op) < 1:
                            continue
                        ev[op_costs[op[0]]] += 1
                        if op.startswith('s<tr>'):
                            trs = op[5:].split('<sep>')
                            if trs[0] in kb_map and trs[1] not in kb_map[trs[0]]:
                                flag = False
                                break
                    if flag:
                        sug_list.append(ed)
                        freqs[ed] = trace[0] * 1.0 / sp.total_word_count
                        evs[ed] = ev

                if term in ngrams:
                    for pred, _ in ngrams[term]:
                        if pred in sug_list:
                            if pred == label:
                                fea_out.write('1 ')
                            else:
                                fea_out.write('0 ')

                            last2v = get_we(w2v, last2)
                            lastv = get_we(w2v, last)
                            candv = get_we(w2v, pred)
                            freq = freqs[pred]
                            bi_freq = get_bifreq(bigrams, last2, pred)
                            ev = evs[pred]
                            if pred.startswith(last) or last.startswith(pred):
                                is_prefix = 1
                            else:
                                is_prefix = 0

                            fea_out.write(get_feature_we(last2v, lastv, candv, freq, bi_freq, ev, is_prefix) + '\n')
