import operator
import time
import os

from pynlpl.textprocessors import *

import numpy as np
import cPickle as pk


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


def last_as_label():
    with open('raw/train.txt', 'r') as fin:
        with open('raw/last_as_label', 'w') as fout:
            count_total = 0
            count_match = 0

            for line in fin:
                line = line.strip()
                index = line.rfind(' ')
                label = line[index + 1:].lower()
                line = line[0:index]
                index = line.rfind(' ')
                if index == -1:
                    last = line.lower()
                else:
                    last = line[index + 1:].lower()

                count_total += 1
                if last == label:
                    count_match += 1
                else:
                    fout.write(str(count_total) + '\t' + last + '\t' + label + '\n')

            print count_match, count_total, count_match * 1.0 / count_total


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
            for word, freq in sorted(dict.items(), key=operator.itemgetter(1), reverse=True):
                fout.write(word + '\t' + str(freq) + '\n')


def count_ngram(N, fin, fout):
    with open(fin, 'r') as corpus_in:
        ngrams = {}
        for line in corpus_in:
            words = tokenize(line)
            for ngram in Windower(words, N, '<s>', None):
                key = '\t'.join(ngram)
                if key in ngrams:
                    ngrams[key] += 1
                else:
                    ngrams[key] = 1

        with open(fout, 'w') as ngrams_out:
            for key, value in sorted(ngrams.items(), key=operator.itemgetter(0), reverse=True):
                ngrams_out.write(key + '\t' + str(value) + '\n')


def test_symspell_big():
    import symspell_python as sp

    sp.init('testdata/big.txt')

    with open('raw/train.txt', 'r') as train_in:
        with open('raw/big_symspell', 'w') as train_out:
            count_total = 0
            count_match = 0
            count_contain = 0
            stime = time.time()

            for line in train_in:
                line = line.strip().lower()
                index = line.rfind(' ')
                label = line[index + 1:]
                line = line[:index]
                index = line.rfind(' ')
                last = line[index + 1:]

                count_total += 1
                if count_total % 10000 == 0:
                    etime = time.time()
                    print count_total, etime - stime
                    stime = etime

                sug_list = sp.get_suggestions(last, True)
                if len(sug_list) < 1:
                    continue

                if label == sug_list[0][0]:
                    count_match += 1
                elif label in [x[0] for x in sug_list]:
                    count_contain += 1
                    train_out.write(
                            str(count_total) + '\t' + last + '\t' + label + '\t' + str(len(sug_list)) + '\t' + str(
                                    filter(lambda x: x[0] == label, sug_list)[0]) + '\n')

            print count_match, count_contain, count_total
            print count_match * 1.0 / count_total, count_contain * 1.0 / count_total


def test_edit(corpus, train_set):
    import symspell_python as sp
    sp.init(corpus)

    print 'test edit', corpus, train_set
    with open(train_set, 'r') as train_in:
        with open(train_set + '.log', 'w') as log_out:
            count_total = 0
            count_hit = 0
            count_match = 0
            count_match_prefix = 0
            count_match_edit = 0
            count_contain = 0
            stime = time.time()

            for line in train_in:
                line = line.strip().lower()
                index = line.rfind(' ')
                label = line[index + 1:]
                line = line[:index]
                index = line.rfind(' ')
                last = line[index + 1:]

                count_total += 1
                # if count_total == 100:
                #     exit(0)

                if count_total % 10000 == 0:
                    etime = time.time()
                    print count_total, etime - stime
                    stime = etime

                sug_list = sp.get_suggestions(last, True)
                if len(sug_list) < 1:
                    continue

                count_hit += 1

                completions = filter(lambda x: x[0].startswith(last), sug_list)
                if len(completions) > 0:
                    if label == completions[0][0]:
                        count_match += 1
                        count_match_prefix += 1
                    elif label == sug_list[0][0]:
                        count_match += 1
                        count_match_edit += 1
                    elif label in [x[0] for x in sug_list] or label in [x[0] for x in completions]:
                        count_contain += 1
                        log_out.write(str(count_total) + '\t' + last + '\t' + label + '\t' + str(sug_list) + '\n')
                elif label == sug_list[0][0]:
                    count_match += 1
                    count_match_edit += 1
                elif label in [x[0] for x in sug_list]:
                    count_contain += 1
                    log_out.write(str(count_total) + '\t' + last + '\t' + label + '\t' + str(sug_list) + '\n')

        print count_match_prefix, count_match_edit
        print count_match, count_contain, count_hit, count_total
        print 'match rate:', count_match * 1.0 / count_total
        print 'contain rate:', count_contain * 1.0 / count_total
        print 'hit rate:', count_hit * 1.0 / count_total


def load_bigrams(bigram_path):
    bigrams_bin_path = bigram_path + '.bin'
    if os.path.isfile(bigrams_bin_path):
        return pk.load(open(bigrams_bin_path, 'r'))

    bigrams = {}
    with open(bigram_path, 'r') as bigram_in:
        for line in bigram_in:
            arr = line.strip().split()
            if arr[0] in bigrams:
                bigrams[arr[0]].append((arr[1], int(arr[2])))
            else:
                bigrams[arr[0]] = [(arr[1], int(arr[2]))]

    for key in bigrams.keys():
        bigrams[key].sort(reverse=True, key=operator.itemgetter(1))

    pk.dump(bigrams, open(bigrams_bin_path, 'w'))
    return bigrams


def test_bigram(train_set, corpus):
    import symspell_python as sp

    print 'test bigram'
    print 'train set', train_set
    print 'corpus', corpus

    sp.init(corpus)
    with open(train_set, 'r') as train_in:
        bigrams = load_bigrams(corpus + '.bigrams')

        count_total = 0
        count_hit = 0
        count_match = 0
        count_contain = 0
        count_best_edit = 0
        stime = time.time()
        # sp.verbose = 0

        for line in train_in:
            line = line.strip().lower()
            index = line.rfind(' ')
            label = line[index + 1:]
            line = line[:index]
            index = line.rfind(' ')
            last = line[index + 1:]
            line = line[:index]
            index = line.rfind(' ')
            if index != -1:
                last2 = line[index + 1:]
            else:
                last2 = '<s>'

            count_total += 1

            # only bigram
            # if last2 in bigrams:
            #     count_hit += 1
            #     cand = bigrams[last2]
            #     # max_freq = cand[0][1]
            #     best_cand = cand[0][0]
            #
            #     if best_cand == label:
            #         count_match += 1

            sug_list = sp.get_suggestions(last, True)
            if len(sug_list) < 1:
                continue
            count_hit += 1

            # only symspell
            # if label == sug_list[0][0]:
            #     count_match += 1
            # elif label in [x[0] for x in sug_list]:
            #     count_contain += 1

            if sug_list[0][1] == 0:
                count_best_edit += 1
                if sug_list[0][0] == label:
                    count_match += 1

            # bigram(symspell)
            # if last2 in bigrams:
            #     cand = {}
            #     for word, freq in bigrams[last2]:
            #         cand[word] = freq
            #     max_freq = -1
            #     best_edit = ''
            #     for word, freq in sug_list:
            #         if word in cand and max_freq < cand[word]:
            #             max_freq = cand[word]
            #             best_edit = word
            #     if best_edit == label:
            #         count_match += 1

            if count_total % 1000 == 0:
                etime = time.time()
                print count_total, etime - stime
                stime = etime

        # print count_match, count_contain, count_hit, count_total
        # print count_match * 1.0 / count_hit, count_match * 1.0 / count_total
        # print count_hit * 1.0 / count_total
        print 'check best_edit vs label'
        print count_match, count_best_edit, count_hit, count_total
        print count_match * 1.0 / count_best_edit, count_match * 1.0 / count_hit, count_match * 1.0 / count_total


if __name__ == "__main__":
    # test_ngram_limit('raw/corpus.sens.10k', 'raw/train')

    # load_bigrams('raw/corpus.sens.10k.bigrams')
    # split_train('raw/train', 0.1)
    # test_bigram('raw/train.part1', 'raw/corpus.sens.10k')
    # clean_sens('raw/corpus.words.15k', 'raw/corpus.sens.15k')
    test_edit('raw/corpus.sens.14k', 'raw/train.part1')
