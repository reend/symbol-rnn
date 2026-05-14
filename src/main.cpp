#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>

std::vector<float> one_hot(int idx, int vocab_size) {
    std::vector<float> v(vocab_size, 0.0f);
    v[idx] = 1.0f;
    return v;
}

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

std::vector<float> matvec(const Matrix& m, const std::vector<float>& v) {
    std::vector<float> out(m.rows, 0.0f);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++)
            out[r] += m.at(r, c) * v[c];
    return out;
}

std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (int i = 0; i < a.size(); i++) {
        out[i] = a[i] + b[i];
    }
    return out;
}

std::vector<float> tanh_vec(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    for (int i = 0; i < v.size(); i++) {
        out[i] = std::tanh(v[i]);
    }
    return out;
}

std::vector<float> softmax(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    float max_v = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (int i = 0; i < v.size(); i++) {
        out[i] = std::exp(v[i] - max_v);
        sum += out[i];
    }
    for (float &x : out) x /= sum;
    return out;
}

float cross_entropy(const std::vector<float>& probs, int target) {
    return -std::log(probs[target] + 1e-8f);
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

    // W initialization

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

    // forward pass

    const int seq_len = 25;

    std::vector<float> h(hidden_size, 0.0f);

    std::vector<std::vector<float>> xs, hs, ps;
    hs.push_back(h);

    float total_loss = 0.0f;

    for (int t = 0; t < seq_len; t++) {

        std::vector<float> x = one_hot(data[t], vocab_size);

        std::vector<float> a1 = matvec(Wxh, x);
        std::vector<float> a2 = matvec(Whh, hs.back());
        std::vector<float> a3 = add(add(a1, a2), bh.data);
        std::vector<float> h1 = tanh_vec(a3);

        std::vector<float> y = add(matvec(Why, h1), by.data);
        std::vector<float> p = softmax(y);

        xs.push_back(x);
        hs.push_back(h1);
        ps.push_back(p);

       total_loss += cross_entropy(p, data[t + 1]);
    }

    std::cout << "loss: " << total_loss / seq_len << "\n";

    return 0;
}