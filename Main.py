import xgboost as xgb

from Reader import *


def last_as_label(train_set):
    print 'test last as label', train_set
    with open(train_set, 'r') as fin:
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

        print count_match, count_total, count_match * 1.0 / count_total


def test_edit(corpus, train_set):
    from SymSpell import SymSpell
    sp = SymSpell(corpus)
    sp.max_edit_distance = 1

    print 'test edit', corpus, train_set
    with open(train_set, 'r') as train_in:
        count_total = 0
        count_match1 = 0
        count_match2 = 0
        count_contain1 = 0
        count_contain2 = 0
        count_total1 = 0
        count_total2 = 0

        stime = time.time()

        for line in train_in:
            line = line.strip().lower()
            index1 = line.rfind(' ')
            label = line[index1 + 1:]
            index2 = line.rfind(' ', 0, index1)
            last = line[index2 + 1:index1]

            count_total += 1

            if count_total % 10000 == 0:
                etime = time.time()
                print count_total, etime - stime
                stime = etime

            sug_list = sp.get_suggestions(last, True)

            if sp.get_index(last) == 0:
                count_total1 += 1
                if len(sug_list) < 1:
                    continue

                if sug_list[0][0] == label:
                    count_match1 += 1
                elif label in [x[0] for x in sug_list]:
                    count_contain1 += 1
            else:
                count_total2 += 1
                if len(sug_list) < 1:
                    continue

                if sug_list[0][0] == label:
                    count_match2 += 1
                elif label in [x[0] for x in sug_list]:
                    count_contain2 += 1

            # completion
            # completions = filter(lambda x: x[0].startswith(last), sug_list)
            # if len(completions) > 0:
            #     if label == completions[0][0]:
            #         count_match += 1
            #     # elif label == sug_list[0][0]:
            #     #     count_match += 1
            #     elif label in [x[0] for x in sug_list] or label in [x[0] for x in completions]:
            #         count_contain += 1
            #         # elif label == sug_list[0][0]:
            #         #     count_match += 1
            #         # count_match_edit += 1
            # elif label in [x[0] for x in sug_list]:
            #     count_contain += 1


    print count_match1, count_contain1, count_match2, count_contain2, count_total1, count_total2, count_total
    print 'match rate:', count_match1 * 1.0 / count_total1, count_match2 * 1.0 / count_total2
    print 'overall match rate', count_match1 * 1.0 / count_total, count_match2 * 1.0 / count_total
    print 'contain rate:', count_contain1 * 1.0 / count_total1, count_contain2 * 1.0 / count_total2
    print count_total1 * 1.0 / count_total


def test_vector(corpus, train_set, model_path):
    from SymSpell import SymSpell
    sp = SymSpell(corpus)
    print 'test vector', corpus, train_set, model_path

    with open(train_set, 'r') as train_in:
        # with open()
        bst = xgb.Booster(model_file=model_path)
        bigrams = load_bigrams2(corpus + '.bigram')
        count_total = 0
        count_match = 0
        stime = time.time()

        for line in train_in:
            line = line.strip().lower()
            index1 = line.rfind(' ')
            label = line[index1 + 1:]
            index2 = line.rfind(' ', 0, index1)
            last = line[index2 + 1:index1]
            index3 = line.rfind(' ', 0, index2)
            if index3 != -1:
                last2 = line[index3 + 1: index2]
            else:
                last2 = '<s>'

            count_total += 1
            if count_total % 1000 == 0:
                etime = time.time()
                print count_total, etime - stime
                stime = etime

            sug_list = sp.get_suggestions(last, True)
            if len(sug_list) == 0:
                continue

            # if sp.get_index(last) == 0:
            foos = [get_feature_word_vector(last2, last, edit, label, sp.get_freq(edit),
                                            get_bifreq(bigrams, last2, edit)) for edit, _ in sug_list]
            dtest = xgb.DMatrix(foo_2_csr(foos), silent=True)
            preds = bst.predict(dtest)
            best_cand = sug_list[np.argmax(preds)][0]
            if best_cand == label:
                count_match += 1
                # elif sug_list[0][0] == label:
                #     count_match += 1

    print count_match, count_total
    print count_match * 1.0 / count_total


def test_bigram(corpus, train_set):
    print 'test bigram'
    print 'train set', train_set
    print 'corpus', corpus

    from SymSpell import SymSpell
    sp = SymSpell(corpus)
    with open(train_set, 'r') as train_in:
        bigrams = load_bigrams2(corpus + '.bigram')

        count_total = 0
        count_match = 0
        stime = time.time()
        # sp.verbose = 0

        for line in train_in:
            line = line.strip().lower()
            index1 = line.rfind(' ')
            label = line[index1 + 1:]
            index2 = line.rfind(' ', 0, index1)
            last = line[index2 + 1:index1]
            index3 = line.rfind(' ', 0, index2)
            if index3 != -1:
                last2 = line[index3 + 1: index2]
            else:
                last2 = '<s>'

            count_total += 1

            if last2 in bigrams:
                cand = bigrams[last2]
                sug_list = sp.get_suggestions(last, True)
                if len(sug_list) > 0:
                    min_edit_dist = 4
                    max_freq = 0
                    best_cand = ''
                    for edit, val in sug_list:
                        if edit in cand:
                            if val[1] < min_edit_dist:
                                min_edit_dist = val[1]
                                best_cand = edit
                            elif val[1] == min_edit_dist and val[0] > max_freq:
                                max_freq = val[0]
                                best_cand = edit
                    if best_cand == label:
                        count_match += 1

            if count_total % 10000 == 0:
                etime = time.time()
                print count_total, etime - stime
                stime = etime

        print 'with max freq'
        print count_match, count_total
        print count_match * 1.0 / count_total


if __name__ == "__main__":
    # make_feature('raw/corpus.sens.14k', 'raw/train.part1.part1')
    # split_train('raw/train.part1.vfeature2', 0.7)
    # make_feature_word_vector('raw/corpus.sens.14k', 'raw/train.part1')
    # test_bigram('raw/corpus.sens.14k', 'raw/train.part1')
    # test_vector('raw/corpus.sens.14k', 'raw/train.part1', 'raw/train.part1.vfeature2.part2.model')
    # last_as_label('raw/train.part1')
    test_edit('raw/corpus.sens.14k', 'raw/train.part1')
