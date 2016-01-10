import gensim
import logging


class MySens:
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname):
            yield line.strip().lower().split()


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sens = MySens('raw/corpus.sens')
model = gensim.models.Word2Vec(sens)
