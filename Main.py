from Reader import *


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


def test_edit(corpus, train_set):
    from SymSpell import SymSpell
    sp = SymSpell(corpus)

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


def test_bigram(train_set, corpus):
    print 'test bigram'
    print 'train set', train_set
    print 'corpus', corpus

    from SymSpell import SymSpell
    sp = SymSpell(corpus)
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
    # test_edit('raw/corpus.sens.14k', 'raw/train.part1')

    # count_ngram(2, 'raw/corpus.sens.14k', 'raw/corpus.sens.14k.bigram')

    # from SymSpell import SymSpell
    # sp = SymSpell('raw/corpus.sens.14k')
    # make_feature('raw/corpus.sens.14k', 'raw/train.part1.part1')
    split_train('raw/train.part1.feature', 0.7)
