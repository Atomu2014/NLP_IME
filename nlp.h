#ifndef NLP_H
#define NLP_H

#include <vector>
#include <string>
#include <sstream>

using namespace std;

string i_2_a(int n){
	stringstream ss;
	string str;
	ss << n;
	ss >> str;
	return str;
}

int a_2_i(char * a){
	stringstream ss;
	int n;
	ss << a;
	ss >> n;
	return n;
}

string join(string seq, vector<string> words, int lb, int rb){
	string res = words[lb];
	for (int i=lb+1; i<=rb; ++i){
		res += seq + words[i];
	}
	return res;
}

string lower(string str){
	string res = "";
	for (int i=0; i<str.length(); ++i){
		if (str[i] >= 'A' && str[i] <= 'Z'){
			res += str[i] + 'a' - 'A';
		} else {
			res += str[i];
		}
	}
	return res;
}

string trim(string str){
	int i=0, j=str.length()-1;
	while (i < str.length() && str[i] == '\''){
		i++;
	}
	while (j >= i && str[j] == '\''){
		j--;
	}

	if (i <= j){
		return str.substr(i, j-i+1);
	} else {
		return "";
	}
}

bool is_ascii(string line){
	for (int i=0; i<line.length(); ++i){
		// cout << int(line[i]) << ' ';
		if (!isascii(line[i])){
			return false;
		}
	}
	return true;
}

bool is_word(string str){
	for (int i=0; i<str.length(); ++i){
		if (str[i] != '\'' && (str[i] < 'a' || str[i] > 'z')){
			return false;
		}
	}
	return str.length() > 1 || (str.length() == 1 && str[0] != '\'');
}

vector<string> split_words(string line){
    vector<string> arr;

    size_t found = 0;
    size_t from = 0;

    string word;

    while ((found = line.find(' ', from)) != string::npos || (found = line.find('.', from)) != string::npos){
        word = line.substr(from, found-from);
        if (word.length() > 0){
        	arr.push_back(word);
        }
        from = found+1;
    }

    arr.push_back(line.substr(from, line.length()-from));

    return arr;
}

vector<string> split(string line, string sep){
    vector<string> arr;

    size_t found = 0;
    size_t from = 0;

    string word;

    while ((found = line.find(sep, from)) != string::npos){
        word = line.substr(from, found-from);
        if (word.length() > 0){
        	arr.push_back(word);
        }
        from = found+sep.length();
    }

    arr.push_back(line.substr(from, line.length()-from));

    return arr;
}

string replace(string line, string a, string b){
	vector<string> strs = split(line, a);
	return join(b, strs, 0, strs.size()-1);
}

#endif