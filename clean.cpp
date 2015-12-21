#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include "nlp.h"

using namespace std;

int main(){
	ifstream corpus_in("raw/english.corpus");
	ofstream corpus_out("english.corpus.clean");

	string line;
	int nline = 0;
	string buffer;

	vector<string> words;
	string word;

	while (getline(corpus_in, line)){
		nline++;

		if (is_ascii(line)){
			line = replace(line, "\'\'", " ");
			words = split_words(line);
			buffer = "";

			for (int i=0; i<words.size(); ++i){
				word = lower(trim(words[i]));

				if (is_word(word)){
					if (buffer.length() > 0){
						buffer += ' ' + word;					
					} else {
						buffer += word;
					}
				}
			}

			if (buffer.length() > 0){
				corpus_out << buffer << '\n';
			}
		}
	}

	return 0;
}