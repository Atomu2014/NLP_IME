'''
symspell_python.py

################

To run, execute python symspell_python.py at the prompt.
Make sure the dictionary "big.txt" is in the current working directory.
Enter word to correct when prompted.

################

v 1.1 last revised 28 Nov 2015
For further info, check out spark-n-spell.com
or e-mail mail@k-lo.ca

################

This program is a Python version of a spellchecker based on SymSpell, 
a Symmetric Delete spelling correction algorithm developed by Wolf Garbe 
and originally written in C#.

From the original SymSpell documentation:

"The Symmetric Delete spelling correction algorithm reduces the complexity 
 of edit candidate generation and dictionary lookup for a given Damerau-
 Levenshtein distance. It is six orders of magnitude faster and language 
 independent. Opposite to other algorithms only deletes are required, 
 no transposes + replaces + inserts. Transposes + replaces + inserts of the 
 input term are transformed into deletes of the dictionary term.
 Replaces and inserts are expensive and language dependent: 
 e.g. Chinese has 70,000 Unicode Han characters!"

For further information on SymSpell, please consult the original
documentation:
  URL: blog.faroo.com/2012/06/07/improved-edit-distance-based-spelling-correction/
  Description: blog.faroo.com/2012/06/07/improved-edit-distance-based-spelling-correction/

The current version of this program will output all possible suggestions for
corrections up to an edit distance (configurable) of max_edit_distance = 3. 

With the exception of the use of a third-party method for calculating
Demerau-Levenshtein distance between two strings, we have largely followed 
the structure and spirit of the original SymSpell algorithm and have not 
introduced any major optimizations or improvements.

Changes in this version (1.1):
We implement allowing for less verbose options: e.g. when only a single
recommended correction is required, the search may terminate early, thereby
enhancing performance. 

#################

Sample output:

Please wait...
Creating dictionary...
total words processed: 1105285
total unique words in corpus: 29157
total items in dictionary (corpus words and deletions): 2151998
  edit distance for deletions: 3
  length of longest word in corpus: 18
 
Word correction
---------------
Enter your input (or enter to exit): there
('there', (2972, 0))
 
Enter your input (or enter to exit): hellot
('hello', (1, 1))
 
Enter your input (or enter to exit): accomodation
('accommodation', (5, 1))
 
Enter your input (or enter to exit): 
goodbye


'''

import cPickle as pk
import os
import re
import time

from Editor import *


class SymSpell:
    def __init__(self, corpus_path, max_dist=3, add_dict=None):
        self.max_edit_distance = max_dist
        self.verbose = 2
        # 0: top suggestion
        # 1: all suggestions of smallest edit distance
        # 2: all suggestions <= max_edit_distance (slower, no early termination)

        self.dictionary = {}
        self.longest_word_length = 0
        self.total_word_count = 0
        self.unique_word_count = 0

        if add_dict:
            self.add_dict = add_dict
            self.add_dict_size = len(add_dict)

        print "Please wait..."
        start_time = time.time()
        dict_path = corpus_path + '.bin'
        if os.path.isfile(dict_path):
            [self.longest_word_length, self.total_word_count, self.unique_word_count, self.dictionary] = pk.load(
                    open(dict_path, 'r'))
        else:
            self.create_dictionary(corpus_path, False)
            pk.dump([self.longest_word_length, self.total_word_count, self.unique_word_count, self.dictionary],
                    open(dict_path, 'w'))
        run_time = time.time() - start_time
        print "total words processed: %i" % self.total_word_count
        print "total unique words in corpus: %i" % self.unique_word_count
        print "total items in dictionary (corpus words and deletions): %i" % len(self.dictionary)
        print "  edit distance for deletions: %i" % self.max_edit_distance
        print "  length of longest word in corpus: %i" % self.longest_word_length
        print '-----'
        print '%.2f seconds to run' % run_time
        print '-----'
        self.test()

    def get_deletes_list(self, w):
        '''
        given a word, derive strings with up to max_edit_distance characters
        deleted
        '''

        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        '''add word and its derived deletions to dictionary'''
        # check if word is already in dictionary
        # dictionary entries are in the form: (list of suggested corrections,
        # frequency of word in corpus)
        new_real_word_added = False
        if w in self.dictionary:
            # increment count of word in corpus
            self.dictionary[w] = [self.dictionary[w][0], self.dictionary[w][1] + 1]
        else:
            self.dictionary[w] = [[], 1]
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            # first appearance of word in corpus
            # n.b. word may already be in dictionary as a derived word
            # (deleting character from a real word)
            # but counter of frequency of word in corpus is not incremented
            # in those cases)
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    # add (correct) word to delete's suggested correction list
                    # if not already there
                    if item not in self.dictionary[item][0]:
                        self.dictionary[item][0].append(w)
                else:
                    # note frequency of word in corpus is not incremented
                    self.dictionary[item] = [[w], 0]

        return new_real_word_added

    def create_dictionary(self, fname, silent=True):
        self.total_word_count = 0
        self.unique_word_count = 0

        with open(fname) as file:
            print "Creating dictionary..."
            nline = 0
            stime = time.time()
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall("[a-z']+", line.lower())
                for word in words:
                    self.total_word_count += 1
                    if self.create_dictionary_entry(word):
                        self.unique_word_count += 1

                if not silent:
                    nline += 1
                    if nline % 100000 == 0:
                        etime = time.time()
                        print nline, etime - stime
                        stime = etime

        index = 0
        for k in self.dictionary.keys():
            if self.dictionary[k][1] > 0:
                index += 1
                self.dictionary[k].append(index)
        print index, 'unique words'

    def get_suggestions(self, string, silent=False):
        '''
        return list of suggested corrections for potentially incorrectly
        spelled word
        '''
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print "no items in dictionary within maximum edit distance"
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {}  # items other than string that we've checked

        while len(queue) > 0:
            q_item = queue[0]  # pop
            queue = queue[1:]

            # early exit
            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            # process queue item
            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if (self.dictionary[q_item][1] > 0):
                    # word is in dictionary, and is a word from the corpus, and
                    # not already in suggestion list so add to suggestion
                    # dictionary, indexed by the word with value (frequency in
                    # corpus, edit distance)
                    # note q_items that are not the input string are shorter
                    # than input string since only deletes are added (unless
                    # manual dictionary corrections are added)
                    assert len(string) >= len(q_item)
                    '''
                    suggest_dict[q_item] = (self.dictionary[q_item][1], len(string) - len(q_item))
                    '''
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item), 'a' * (len(string) - len(q_item)))
                    # early exit
                    if ((self.verbose < 2) and (len(string) == len(q_item))):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                # the suggested corrections for q_item as stored in
                # dictionary (whether or not q_item itself is a valid word
                # or merely a delete) can be valid corrections
                for sc_item in self.dictionary[q_item][0]:
                    if (sc_item not in suggest_dict):

                        # compute edit distance
                        # suggested items should always be longer
                        # (unless manual corrections are added)
                        assert len(sc_item) > len(q_item)

                        # q_items that are not input should be shorter
                        # than original string
                        # (unless manual corrections added)
                        assert len(q_item) <= len(string)

                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        # item in suggestions list should not be the same as
                        # the string itself
                        assert sc_item != string

                        # calculate edit distance using, for example,
                        # Damerau-Levenshtein distance
                        '''
                        item_dist = dameraulevenshtein(sc_item, string)
                        '''
                        item_dist = dldist_with_op(sc_item, string)

                        # do not add words with greater edit distance if
                        # verbose setting not on
                        '''
                        if ((self.verbose < 2) and (item_dist > min_suggest_len)):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist
                        '''
                        if ((self.verbose < 2) and (item_dist[0] > min_suggest_len)):
                            pass
                        elif item_dist[0] <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist[0], item_dist[1])
                            if item_dist[0] < min_suggest_len:
                                min_suggest_len = item_dist[0]

                        # depending on order words are processed, some words
                        # with different edit distances may be entered into
                        # suggestions; trim suggestion dictionary if verbose
                        # setting not on
                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            # now generate deletes (e.g. a substring of string or of a delete)
            # from the queue item
            # as additional items to check -- add to end of queue
            assert len(string) >= len(q_item)

            # do not add words with greater edit distance if verbose setting
            # is not on
            if ((self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len)):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None  # arbitrary value, just to identify we checked this

        # queue is now empty: convert suggestions in dictionary to
        # list for output
        if not silent and self.verbose != 0:
            print "number of possible corrections: %i" % len(suggest_dict)
            print "  edit distance for deletions: %i" % self.max_edit_distance

        # output option 1
        # sort results by ascending order of edit distance and descending
        # order of frequency
        #     and return list of suggested word corrections only:
        # return sorted(suggest_dict, key = lambda x:
        #               (suggest_dict[x][1], -suggest_dict[x][0]))

        # output option 2
        # return list of suggestions with (correction,
        #                                  (frequency in corpus, edit distance)):
        as_list = suggest_dict.items()
        outlist = sorted(as_list, key=lambda (term, (freq, dist, op)): (dist, -freq))

        if self.verbose == 0:
            if len(outlist) > 0:
                return [outlist[0]]
            else:
                return []
        else:
            return outlist

    def test(self):
        counter = 0
        for k in self.dictionary.keys():
            # print k, self.dictionary[k]
            if len(self.dictionary[k]) > 2:
                counter += 1
        print counter

    def get_index(self, word):
        if word in self.dictionary and len(self.dictionary[word]) > 2:
            return self.dictionary[word][2]
        return 0

    def get_index2(self, word):
        if word in self.dictionary and len(self.dictionary[word]) > 2:
            return self.dictionary[word][2]
        if word in self.add_dict:
            return self.add_dict[word]
        return 0

    def get_freq(self, word, norm=True):
        if word in self.dictionary:
            if norm:
                return self.dictionary[word][1] * 1.0 / self.total_word_count
            else:
                return self.dictionary[word][1]
        return 0


if __name__ == "__main__":
    sym = SymSpell('raw/corpus.sens.14k')

    print " "
    print "Word correction"
    print "---------------"

    while True:
        word_in = raw_input('Enter your input (or enter to exit): ')
        if len(word_in) == 0:
            print "goodbye"
            break
        start_time = time.time()
        print sym.get_suggestions(word_in)
        run_time = time.time() - start_time
        print '-----'
        print '%.5f seconds to run' % run_time
        print '-----'
        print " "
