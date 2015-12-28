import xgboost as xgb

from Reader import *


def test_label(train_set):
    print 'test last as label', train_set
    with open(train_set, 'r') as fin:
        count_total = 0
        count_match = 0

        for line in fin:
            label, last = split_last_words(line, 2)

            count_total += 1
            if last == label:
                count_match += 1

        print count_match, count_total, count_match * 1.0 / count_total


def test_bigram(corpus, train_set):
    print 'test bigram', train_set, corpus
    with open(train_set, 'r') as train_in:
        bigrams = load_bigrams(corpus + '.bigram')

        count_total = 0
        count_contain = 0
        count_pos = 0
        count_neg = 0
        stime = time.time()

        for line in train_in:
            label, last, last2 = split_last_words(line, 3)

            count_total += 1

            if last2 in bigrams:
                cands = set([x[0] for x in bigrams[last2]])
                if label in cands:
                    count_contain += 1
                    count_pos += 1
                    count_neg += len(cands) - 1

            if count_total % 10000 == 0:
                etime = time.time()
                print count_total, etime - stime
                stime = etime

        print count_contain, count_total
        print count_contain * 1.0 / count_total
        print count_pos, count_neg, count_neg * 1.0 / count_pos


def test_edit(corpus, train_set):
    from SymSpell import SymSpell
    sp = SymSpell(corpus, 2)

    print 'test edit', corpus, train_set
    with open(train_set, 'r') as train_in:
        count_total = 0
        count_contain = 0
        count_pos = 0
        count_neg = 0

        stime = time.time()

        for line in train_in:
            label, last = split_last_words(line.strip().lower(), 2)

            count_total += 1

            if count_total % 10000 == 0:
                etime = time.time()
                print count_total, etime - stime
                stime = etime

            sug_list = sp.get_suggestions(last, True)
            if label in [x[0] for x in sug_list]:
                count_pos += 1
                count_contain += 1
                count_neg += len(sug_list) - 1

    print count_contain, count_total
    print 'contain rate', count_contain * 1.0 / count_total
    print 'pos neg', count_pos, count_neg, count_neg * 1.0 / count_pos


def test_word_vector(corpus, train_set, model_path):
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
            label, last, last2 = split_last_words(line, 3)

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


def test_edit_vector(corpus, train_set, model_path):
    print 'test edit vector', corpus, train_set, model_path

    from SymSpell import SymSpell
    sp = SymSpell(corpus, 1, load_additional_dict(corpus, 'raw/train'))
    with open(train_set, 'r') as train_in:
        # with open(train_set + '.r', 'w') as logr_out:
        #     with open(train_set + '.w', 'w') as logw_out:
                bst = xgb.Booster(model_file=model_path)

                count_total = 0
                count_match = 0
                stime = time.time()

                for line in train_in:
                    line = line.strip().lower()
                    label, last, last2 = split_last_words(line, 3)
                    count_total += 1
                    if count_total % 10000 == 0:
                        etime = time.time()
                        print count_total, etime - stime, count_match * 1.0 / count_total
                        stime = etime

                    sug_list = sp.get_suggestions(last, True)

                    if len(sug_list) == 0:
                        continue

                    foos = [get_feature_edit_vector(sp, last2, last, edit, val[2], 1) for edit, val in sug_list]
                    dtest = xgb.DMatrix(foo_2_csr(foos), silent=True)
                    preds = bst.predict(dtest)
                    best_cand = sug_list[np.argmax(preds)][0]

                    if best_cand == label:
                        count_match += 1

    print count_match, count_total
    print count_match * 1.0 / count_total


if __name__ == "__main__":
    # split_train('raw/train.efeature', 0.7)
    # make_feature('raw/corpus.sens.14k', 'raw/train.part1.part1')
    # make_feature_word_vector('raw/corpus.sens.14k', 'raw/train.part1')
    # make_feature_edit_vector('raw/corpus.sens.14k', 'raw/train.part1', 1)
    # make_feature_edit_vector('raw/corpus.sens.14k', 'raw/train.part2', 1)
    # test_label('raw/train.part1')
    # test_bigram('raw/corpus.sens.14k', 'raw/train.part1')
    # test_edit('raw/corpus.sens.14k', 'raw/train.part1')
    # test_vector('raw/corpus.sens.14k', 'raw/train.part1', 'raw/train.part1.vfeature2.part2.model')
    # test_edit_vector('raw/corpus.sens.14k', 'raw/train.part1', 'raw/train.part2.ellfeature1.model')
    # test_edit_vector('raw/corpus.sens.14k', 'raw/train.part2', 'raw/train.part2.ellfeature1.model')
    print 6
    ngram_on_train('raw/train.part2', 'raw/train.part1', 6)
    print 7
    ngram_on_train('raw/train.part2', 'raw/train.part1', 7)
