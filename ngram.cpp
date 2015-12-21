#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "nlp.h"

using namespace std;

int main(int argc, char * argv[]){
	ifstream corpus_in("english.corpus.clean");

	string line;
	int nline = 0;
	vector<string> tokens;
	int N = a_2_i(argv[1]);
	string ngram;

	map<string, int> dict;

	while (getline(corpus_in, line)){
		tokens = split_words(line);

		nline++;

		for (int i=0; i+N-1<tokens.size(); ++i){
			ngram = join("\t", tokens, i, i+N-1);
		
			if (dict.find(ngram) == dict.end()){
				dict[ngram] = 1;
			} else {
				dict[ngram]++;
			}
		} 
	}

	ofstream ngram_out(argv[2]);

	map<string, int>::iterator mit;
	for (mit = dict.begin(); mit != dict.end(); ++mit){
		if (mit->second > 1){
			ngram_out << mit->first + '\t' + i_2_a(mit->second) + '\n';
		}
	}
}