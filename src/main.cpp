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

Matrix outer(const std::vector<float>& a, const std::vector<float>& b) 
{
    Matrix out(a.size(), b.size());
    for (int i = 0; i < a.size(); i++)
        for (int j = 0; j < b.size(); j++)
            out.at(i, j) = a[i] * b[j];
    return out;
}

void mat_add(Matrix &a, const Matrix &b)
{
    for (int i = 0; i < a.data.size(); i++)
        a.data[i] += b.data[i];
}

std::vector<float> matvec_T(const Matrix& m, const std::vector<float>& v) 
{
    std::vector<float> out(m.cols, 0.0f);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++)
            out[c] += m.at(r, c) * v[r];

    return out;
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
    const float lr    = 0.01f;

    for (int epoch = 0; epoch < 100; epoch++) {

    std::vector<float> h(hidden_size, 0.0f);
    float total_loss = 0.0f;
    int steps = 0;

    for (int pos = 0; pos + seq_len + 1 < (int)data.size(); pos += seq_len) {

    std::vector<std::vector<float>> xs, hs, ps;
    hs.push_back(h);

    for (int t = 0; t < seq_len; t++) {

        std::vector<float> x = one_hot(data[pos + t], vocab_size);

        std::vector<float> a1 = matvec(Wxh, x);
        std::vector<float> a2 = matvec(Whh, hs.back());
        std::vector<float> a3 = add(add(a1, a2), bh.data);
        std::vector<float> h1 = tanh_vec(a3);

        std::vector<float> y = add(matvec(Why, h1), by.data);
        std::vector<float> p = softmax(y);

        xs.push_back(x);
        hs.push_back(h1);
        ps.push_back(p);

       total_loss += cross_entropy(p, data[pos + t + 1]);
    }

    h = hs.back();
    steps++;

    // gradients of matrices

    Matrix dWxh(hidden_size, vocab_size);
    Matrix dWhh(hidden_size, hidden_size);
    Matrix dWhy(vocab_size,  hidden_size);
    Matrix dbh(hidden_size, 1);
    Matrix dby(vocab_size,  1);

    std::vector<float> dh_next(hidden_size, 0.0f);

    for (int t = seq_len - 1; t >= 0; t --) 
    {
        auto dy = ps[t];
        dy[data[pos + t + 1]] -= 1.0f;

        mat_add(dWhy, outer(dy, hs[t + 1]));
        for (int i = 0; i < vocab_size; i++) 
        {
            dby.data[i] += dy[i];
        }

        auto dh = add(matvec_T(Why, dy), dh_next);

        std::vector<float> dh_raw(hidden_size);
        for (int i = 0; i < hidden_size; i++)
            dh_raw[i] = dh[i] * (1.0f - hs[t + 1][i] * hs[t + 1][i]);

        mat_add(dWxh, outer(dh_raw, xs[t]));
        mat_add(dWhh, outer(dh_raw, hs[t]));
        for (int i = 0; i < hidden_size; i++)
            dbh.data[i] += dh_raw[i];

        dh_next = matvec_T(Whh, dh_raw);
    }

    auto clip_and_update = [&](Matrix &W, Matrix &dW)
    {
        for (int i = 0; i < W.data.size(); i++) 
        {
            dW.data[i] = std::max(-5.0f, std::min(5.0f, dW.data[i]));
            W.data[i] -= lr * dW.data[i];
        }
    };

    clip_and_update(Wxh, dWxh);
    clip_and_update(Whh, dWhh);
    clip_and_update(Why, dWhy);
    clip_and_update(bh,  dbh);
    clip_and_update(by,  dby);

    } // end pos

    std::cout << "epoch " << epoch + 1 << " loss: " << total_loss / steps / seq_len << "\n";

    } // end epoch

    return 0;
}