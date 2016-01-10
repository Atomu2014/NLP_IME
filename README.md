# Input Method Engine

## Infrastructure
The whole project contains 6 python modules, recommended environment: Ubuntu 14.04, Python 2.7 64-bit, G++ and R++

- Main.py: the main program
- Editor.py: methods about edit distance
- Reader.py: about file processing, feature making, etc.
- SymSpell.py: extension of Spark-n-Spell module
- Train.py: train logistic regression using XGBoost
- W2V: train Word2Vec models

To run this project, yous should make sure that all these dependencies have been installed (latest version preferred):

- Numpy
- Pickle (or cPickle)
- Pandas
- XGBoost (only required in Train.py)
- Gensim (only required in W2V.py, and some functions in Reader.py)
- Scipy (only required in some functions in Reader.py)
- PyNLPl (only required in some functions in Reder.py)

**Attention**: before running the main program, data files should be put under raw/