#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include "nlp.h"

using namespace std;

int main(){
	ifstream corpus_in("english.corpus.clean");
	ofstream dict_out("dict");

	string line;
	int nline = 0;

	map<string, int> dict;
	vector<string> words;
	string word;

	while (getline(corpus_in, line)){
		nline++;

		words = split(line, ' ');
		for (int i=0; i<words.size(); ++i){
			if (is_words(words[i])){
				word = tolower(words[i]);
				if (dict.find(word) == dict.end()){
					dict[word] = dict.size() + 1;
				}
			}
		}
	}
}