import gensim, logging

from Reader import *


class MySens:
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line.strip().lower().split()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
gensim.models.Word2Vec()
# model = gensim.models.Word2Vec.load('raw/corpus.sens.w2v')
# with open('raw/train', 'r') as fin:
#     count_total = 0
#     count_ex = [0] * 3
#     count_ex2 = [0] * 3
#     count_pre = 0
#     count_re_pre = 0
#     for line in fin:
#         count_total += 1
#         label, last, last2 = split_last_words(line, 3)
#         if label in model:
#             count_ex[0] += 1
#         if last in model:
#             count_ex[1] += 1
#         if last2 in model:
#             count_ex[2] += 1
#         for i in range(len(label), 0, -1):
#             if label[:i] in model:
#                 count_ex2[0] += 1
#                 break
#         for i in range(len(last), 0, -1):
#             if last[:i] in model:
#                 count_ex2[1] += 1
#                 break
#         for i in range(len(last2), 0, -1):
#             if last2[:i] in model:
#                 count_ex2[2] += 1
#                 break
#         if label.startswith(last):
#             count_pre += 1
#
#         if label.startswith(last) or last.startswith(label):
#             count_re_pre += 1
#
#     print count_total
#     print count_ex[0] * 1.0 / count_total
#     print count_ex[1] * 1.0 / count_total
#     print count_ex[2] * 1.0 / count_total
#     print count_ex2[0] * 1.0 / count_total
#     print count_ex2[1] * 1.0 / count_total
#     print count_ex2[2] * 1.0 / count_total
#     print count_pre * 1.0 / count_total
#     print count_re_pre * 1.0 / count_total
