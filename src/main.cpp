#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <set>

int main() {
    std::ifstream file("../data/input.txt");
    std::string text((std::istreambuf_iterator<char>(file)),
                     std::istreambuf_iterator<char>());

    std::set<char> chars(text.begin(), text.end());

        return 0;
}