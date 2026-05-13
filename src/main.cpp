#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <set>

struct Matrix {
    int rows, cols;
    std::vector<float> data;

    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows * cols, 0.0f) {}

    float &at(int r, int c)       { return data[r * cols + c]; }
    float  at(int r, int c) const { return data[r * cols + c]; }
};

void randomize(Matrix& m, float scale) {
    for (float& v : m.data)
        v = ((float)rand() / RAND_MAX * 2 - 1) * scale;
}

std::vector<float> one_hot(int idx, int vocab_size) {
    std::vector<float> v(vocab_size, 0.0f);
    v[idx] = 1.0f;
    return v;
}

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

    std::vector<int> data;
    data.reserve(text.size());

    for (char c : text)
        data.push_back(char_to_idx[c]);

    const int hidden_size = 128;
    const int vocab_size = idx_to_char.size();

    Matrix Wxh(hidden_size, vocab_size);
    Matrix Whh(hidden_size, hidden_size);
    Matrix Why(vocab_size,  hidden_size);
    Matrix bh(hidden_size, 1);
    Matrix by(vocab_size,  1);
    
    srand(42);
    randomize(Wxh, 0.01f);
    randomize(Whh, 0.01f);
    randomize(Why, 0.01f);

    return 0;
}