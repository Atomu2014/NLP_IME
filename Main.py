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


def test_ngram(corpus, ngram_path, train_set):
    print 'test ngram', train_set, corpus
    with open(train_set, 'r') as train_in:
        ngrams = load_ngram3(ngram_path)

        count_total = 0
        count_match = 0
        stime = time.time()

        for line in train_in:
            label, last, last2, last3 = split_last_words(line, 4)

            count_total += 1

            term = '\t'.join([last3, last2])
            if term in ngrams:
                if label == ngrams[term]:
                    count_match += 1

            if count_total % 10000 == 0:
                etime = time.time()
                print count_total, etime - stime
                stime = etime

        print count_match, count_total
        print count_match * 1.0 / count_total


def test_bigram(corpus, ngram_path, train_set):
    print 'test bigram', train_set, corpus
    from SymSpell import SymSpell
    sp = SymSpell(corpus, 3)
    print 'max edit dist', 3
    kb_map = get_kb_map()

    with open(train_set, 'r') as train_in:
        ngrams = load_ngrams(ngram_path)

        count_total = 0
        count_contain = 0
        count_pos = 0
        count_neg = 0
        stime = time.time()

        for line in train_in:
            label, last, last2 = split_last_words(line, 3)

            count_total += 1
            if count_total % 10000 == 0:
                etime = time.time()
                print count_total, etime - stime, count_contain * 1.0 / count_total, count_neg * 1.0 / count_pos
                stime = etime

            term = last2
            edit_list = sp.get_suggestions(last, True, True)
            sug_list = []
            for ed, trace in edit_list:
                flag = True
                for op in trace[2].split('<op>'):
                    if op.startswith('s<tr>'):
                        trs = op[5:].split('<sep>')
                        if trs[0] in kb_map and trs[1] not in kb_map[trs[0]]:
                            flag = False
                            break
                if flag:
                    sug_list.append(ed)

            if term in ngrams:
                cands = []
                for pred, _ in ngrams[term]:
                    if pred in sug_list:
                        cands.append(pred)
                if label in cands:
                    count_contain += 1
                    count_pos += 1
                    count_neg += len(cands) - 1

        print count_contain, count_total
        print count_contain * 1.0 / count_total
        print count_pos, count_neg, count_neg * 1.0 / count_pos


def test_edit(corpus, train_set):
    from SymSpell import SymSpell
    sp = SymSpell(corpus, 1)

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
                print count_total, etime - stime, count_contain * 1.0 / count_total, count_neg * 1.0 / count_pos
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
        bigrams = load_ngrams2(corpus + '.bigram')
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


def test_word_embed(corpus, w2v_path, bigram_path, train_set, model_path):
    print 'test word embed', corpus, w2v_path, bigram_path, train_set, model_path
    from SymSpell import SymSpell
    sp = SymSpell(corpus, 3)
    print 'max edit dist', 3
    from gensim.models.word2vec import Word2Vec
    w2v = Word2Vec.load(w2v_path)
    kb_map = get_kb_map()
    bigrams = load_ngrams2(bigram_path)
    bst = xgb.Booster(model_file=model_path)

    with open(train_set, 'r') as train_in:
        with open(train_set + '.w2v.log', 'w') as log_out:
            count_total = 0
            count_match = 0
            stime = time.time()

            for line in train_in:
                label, last, last2 = split_last_words(line, 3)

                count_total += 1
                if count_total % 10000 == 0:
                    etime = time.time()
                    print count_total, etime - stime, count_match * 1.0 / count_total
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

                if term in bigrams:
                    sug_list = list(set(sug_list).intersection(set([x[0] for x in bigrams[term].items()])))
                    if len(sug_list) < 1:
                        continue
                    foos = [
                        get_feature_we(w2v, last2, last, pred, freqs[pred], get_bifreq(bigrams, last2, pred), evs[pred])
                        for pred in sug_list]
                    dtest = xgb.DMatrix(foo_2_csr(foos), silent=True)
                    preds = bst.predict(dtest)
                    best_cand = sug_list[np.argmax(preds)]

                    if best_cand == label:
                        count_match += 1
                    else:
                        wrong_answer = [(sug_list[i], preds[i]) for i in range(len(sug_list))]
                        wrong_answer.sort(key=lambda x: -x[1])
                        log_out.write(line + '\t' + str(wrong_answer) + '\n')

    print count_total, count_match
    print count_match * 1.0 / count_total


def test_ngram2(corpus, bigram_path, ngram_train_set, test_set, max_len=6, cs=False, eval=False, editor=None):
    # if cs:
    #     ngrams_dict = pk.load(open(ngram_train_set + '.ngram.bin.cs', 'r'))
    # else:
    #     ngrams_dict = pk.load(open(ngram_train_set + '.ngram.bin', 'r'))

    from SymSpell import SymSpell
    sp = SymSpell(corpus, 3, editor=editor)
    print 'max edit dist', 3
    kb_map = get_kb_map()

    with open(test_set, 'r') as test_in:
        # with open(test_set + '.log', 'w') as test_log:
            count_match = 0
            count_miss = 0
            count_total = 0
            stime = time.time()

            for line in test_in:
                    words = line.strip().lower().split('<s>')[-1].split(' ')
                    if words[0] != '<s>':
                        words = ['<s>'] + words
                    if eval:
                        label = split_last_words(line, 1, False)[0]
                        words = words[:-1]

                    last_word = words[-1]
                    count_total += 1
                    if count_total % 1000 == 0:
                        etime = time.time()
                        print count_total, etime - stime, count_match * 1.0 / count_total
                        stime = etime

                # flag = False
                # for i in range(min(max_len, len(words)) - 1, -1, -1):
                #     term = ' '.join(words[len(words) - 1 - i:])
                #     if term in ngrams_dict[i]:
                #         cand = ngrams_dict[i][term]
                #         if words[-2] == '<s>':
                #             cand = cand[0].upper() + cand[1:]
                #         if eval and cand.lower() == label.lower():
                #             count_match += 1
                #         # elif eval and cand.lower() == label.lower():
                #         #     count_case_error += 1
                #         #     test_log.write(line + '\t' + ngrams_dict[i][term] + '\n')
                #         flag = True
                #         break
                # if not flag:
                    edit_list = sp.get_suggestions(last_word, True, True)
                    sug_list = []
                    # freqs = {}
                    # evs = {}
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
                            # freqs[ed] = trace[0] * 1.0 / sp.total_word_count
                            # evs[ed] = ev

                    if len(sug_list) > 0:
                        if sug_list[0].lower() == label.lower():
                            count_match += 1
                        # elif label.lower() in sug_list:
                        #     test_log.write(line.strip() + '\t' + str(sug_list) + '\n')

    print count_match, count_miss, count_total, time.time() - stime
    print count_match * 1.0 / count_total, count_miss * 1.0 / count_total
    return count_match * 1.0 / count_total


if __name__ == "__main__":
    # split_train('raw/train.part1', 0.7)
    # make_feature('raw/corpus.sens.14k', 'raw/train.part1.part1')
    # make_feature_word_vector('raw/corpus.sens.14k', 'raw/train.part1')
    # make_feature_edit_vector('raw/corpus.sens.14k', 'raw/train.part1', 1)
    # make_feature_we('raw/corpus.sens.14k', 'raw/corpus.sens.bigram', 'raw/corpus.sens.w2v.s10', 'raw/train.part1')
    # make_feature_we('raw/corpus.sens.14k', 'raw/corpus.sens.bigram', 'raw/corpus.sens.w2v.s10', 'raw/train.part2')
    # test_label('raw/train.part1')
    # test_ngram('raw/corpus.sens.14k', 'raw/corpus.sens.4gram.max', 'raw/train.part1')
    # test_bigram('raw/corpus.sens.14k', 'raw/corpus.sens.bigram', 'raw/train.part1')
    # test_edit('raw/corpus.sens.14k', 'raw/train.part1')
    # test_vector('raw/corpus.sens.14k', 'raw/train.part1', 'raw/train.part1.vfeature2.part2.model')
    # test_edit_vector('raw/corpus.sens.14k', 'raw/train.part1', 'raw/train.part2.ellfeature1.model')
    # test_edit_vector('raw/corpus.sens.14k', 'raw/train.part2', 'raw/train.part2.ellfeature1.model')
    # test_word_embed('raw/corpus.sens.14k', 'raw/corpus.sens.w2v.s10', 'raw/corpus.sens.bigram', 'raw/train.part1.part2',
    #                 'raw/train.part1.part1.feature.w2v.model')
    # test_word_embed('raw/corpus.sens.14k', 'raw/corpus.sens.w2v.s10', 'raw/corpus.sens.bigram', 'raw/train.part1.part1',
    #                 'raw/train.part1.part1.feature.w2v.model')
    # ngram_on_train('raw/train', 'raw/test_data.txt', 6, False)
    # count_ngram2(6, 'raw/corpus.sens', 'raw/corpus.sens.6gram.max')
    # count_ngram3('raw/train.part2', 6, True)
    # count_ngram3('raw/train.part2', 6, False)

    with open('raw/edit_cost.log2', 'w') as log_out:
        for a in np.arange(1, 2.1, 0.2):
            # for s in np.arange(1, 2.1, 0.2):
                s = 2
                for d in np.arange(1, 2.1, 0.2):
                    for t in np.arange(1, 2.1, 0.2):
                        header = str(a) + '\t' + str(s) + '\t' + str(d) + '\t' + str(t) + '\t'
                        log_out.write(header)
                        score = test_ngram2('raw/corpus.sens.14k', 'raw/corpus.sens.bigram', 'raw/train.part2',
                                            'raw/train.part1.clean', 6, False, True, {'a': a, 's': s, 'd': d, 't': t})
                        log_out.write(str(score) + '\n')
