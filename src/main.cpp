#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <set>

int main() {
    std::ifstream file("data/input.txt");
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

    std::set<char> chars(text.begin(), text.end());

    std::map<char, int> char_to_idx;
    std::vector<char> idx_to_char;

    for (char c : chars) {
        char_to_idx[c] = idx_to_char.size();
        idx_to_char.push_back(c);
    }

    std::cout << "Text: " << text.size() << " symbols\n";
    std::cout << "Vocab: " << idx_to_char.size() << " symbols\n";
    std::cout << "First 200: " << text.substr(0, 200) << "\n";


    return 0;
}