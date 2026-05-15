#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <cblas.h>

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

// version without openblas
std::vector<float> local_matvec(const Matrix& m, const std::vector<float>& v) {
    std::vector<float> out(m.rows, 0.0f);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++)
            out[r] += m.at(r, c) * v[c];
    return out;
}

std::vector<float> matvec(const Matrix& m, const std::vector<float>& v) {
    std::vector<float> out(m.rows, 0.0f);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m.rows, m.cols, 1.0f,
                m.data.data(), m.cols, v.data(), 1, 0.0f, out.data(), 1);
    return out;
}

std::vector<float> add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> out(a.size());
    for (int i = 0; i < a.size(); i++)
        out[i] = a[i] + b[i];
    return out;
}

std::vector<float> tanh_vec(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    for (int i = 0; i < v.size(); i++)
        out[i] = std::tanh(v[i]);
    return out;
}

std::vector<float> rnn_step(
    const std::vector<float>& x,
    const std::vector<float>& h_prev,
    const Matrix& Wxh, const Matrix& Whh, const Matrix& Why,
    const Matrix& bh,  const Matrix& by,
    std::vector<float>& h_out)
{
    std::vector<float> bh_vec(bh.data.begin(), bh.data.end());
    std::vector<float> by_vec(by.data.begin(), by.data.end());

    h_out = tanh_vec(add(add(matvec(Wxh, x), matvec(Whh, h_prev)), bh_vec));
    return add(matvec(Why, h_out), by_vec);
}

std::vector<float> softmax(const std::vector<float>& v) {
    std::vector<float> out(v.size());
    float max_v = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (int i = 0; i < v.size(); i++) {
        out[i] = std::exp(v[i] - max_v);
        sum += out[i];
    }
    for (float& x : out) x /= sum;
    return out;
}

float cross_entropy(const std::vector<float>& probs, int target) {
    return -std::log(probs[target] + 1e-8f);
}

// version without openblas
Matrix local_outer(const std::vector<float>& a, const std::vector<float>& b) {
    Matrix out(a.size(), b.size());
    for (int i = 0; i < a.size(); i++)
        for (int j = 0; j < b.size(); j++)
            out.at(i, j) = a[i] * b[j];
    return out;
}

Matrix outer(const std::vector<float>& a, const std::vector<float>& b) {
    Matrix out(a.size(), b.size());
    cblas_sger(CblasRowMajor, a.size(), b.size(), 1.0f,
               a.data(), 1, b.data(), 1, out.data.data(), b.size());
    return out;
}

// version without openblas
std::vector<float> local_matvec_T(const Matrix& m, const std::vector<float>& v) {
    std::vector<float> out(m.cols, 0.0f);
    for (int r = 0; r < m.rows; r++)
        for (int c = 0; c < m.cols; c++)
            out[c] += m.at(r, c) * v[r];
    return out;
}

std::vector<float> matvec_T(const Matrix& m, const std::vector<float>& v) {
    std::vector<float> out(m.cols, 0.0f);
    cblas_sgemv(CblasRowMajor, CblasTrans, m.rows, m.cols, 1.0f,
                m.data.data(), m.cols, v.data(), 1, 0.0f, out.data(), 1);
    return out;
}

void mat_add(Matrix& a, const Matrix& b) {
    for (int i = 0; i < a.data.size(); i++)
        a.data[i] += b.data[i];
}

void save(const std::string& path, const Matrix& m) {
    std::ofstream f(path, std::ios::binary);
    f.write((char*)m.data.data(), m.data.size() * sizeof(float));
}

void load(const std::string& path, Matrix& m) {
    std::ifstream f(path, std::ios::binary);
    if (f) f.read((char*)m.data.data(), m.data.size() * sizeof(float));
}

std::string generate(
    const Matrix& Wxh, const Matrix& Whh, const Matrix& Why,
    const Matrix& bh, const Matrix& by,
    const std::vector<char>& idx_to_char,
    int start_idx, int hidden_size, int length)
{
    std::vector<float> h(hidden_size, 0.0f);
    auto x = one_hot(start_idx, idx_to_char.size());
    std::string result;
    result += idx_to_char[start_idx];

    for (int i = 0; i < length; i++) {
        std::vector<float> h_next;
        auto y = rnn_step(x, h, Wxh, Whh, Why, bh, by, h_next);
        auto probs = softmax(y);

        float r = (float)rand() / RAND_MAX;
        float cum = 0.0f;
        int idx = probs.size() - 1;
        for (int j = 0; j < probs.size(); j++) {
            cum += probs[j];
            if (r < cum) { idx = j; break; }
        }
        result += idx_to_char[idx];
        x = one_hot(idx, idx_to_char.size());
        h = h_next;
    }
    return result;
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

    load("data/Wxh.bin", Wxh);
    load("data/Whh.bin", Whh);
    load("data/Why.bin", Why);
    load("data/bh.bin",  bh);
    load("data/by.bin",  by);

    const int seq_len   = 25;
    const float lr      = 0.01f;
    int epoch           = 0;

    while (true) {
    std::vector<float> h(hidden_size, 0.0f);

    Matrix dWxh(hidden_size, vocab_size);
    Matrix dWhh(hidden_size, hidden_size);
    Matrix dWhy(vocab_size,  hidden_size);
    Matrix dbh(hidden_size, 1);
    Matrix dby(vocab_size,  1);

    float total_loss = 0.0f;

    std::vector<std::vector<float>> xs, hs, ys, probs_list;
    hs.push_back(h);

    for (int t = 0; t < seq_len; t++) {
        auto x = one_hot(data[epoch * seq_len + t], vocab_size);
        std::vector<float> h_next;
        auto y = rnn_step(x, hs.back(), Wxh, Whh, Why, bh, by, h_next);
        auto p = softmax(y);

        xs.push_back(x);
        hs.push_back(h_next);
        ys.push_back(y);
        probs_list.push_back(p);

        total_loss += cross_entropy(p, data[epoch * seq_len + t + 1]);
    }

    // backprop
    std::vector<float> dh_next(hidden_size, 0.0f);

    for (int t = seq_len - 1; t >= 0; t--) {
        auto dy = probs_list[t];
        dy[data[epoch * seq_len + t + 1]] -= 1.0f;

        mat_add(dWhy, outer(dy, hs[t + 1]));
        for (int i = 0; i < vocab_size; i++) dby.data[i] += dy[i];

        auto dh = add(matvec_T(Why, dy), dh_next);
        std::vector<float> dh_raw(hidden_size);
        for (int i = 0; i < hidden_size; i++)
            dh_raw[i] = dh[i] * (1.0f - hs[t+1][i] * hs[t+1][i]);

        mat_add(dWxh, outer(dh_raw, xs[t]));
        mat_add(dWhh, outer(dh_raw, hs[t]));
        for (int i = 0; i < hidden_size; i++) dbh.data[i] += dh_raw[i];

        dh_next = matvec_T(Whh, dh_raw);
    }

    // gradient clipping + update
    auto clip_and_update = [&](Matrix& W, Matrix& dW) {
        for (int i = 0; i < W.data.size(); i++) {
            dW.data[i] = std::max(-5.0f, std::min(5.0f, dW.data[i]));
            W.data[i] -= lr * dW.data[i];
        }
    };

    clip_and_update(Wxh, dWxh);
    clip_and_update(Whh, dWhh);
    clip_and_update(Why, dWhy);
    clip_and_update(bh,  dbh);
    clip_and_update(by,  dby);

    if (epoch % 100 == 0) {
        std::cout << "epoch " << epoch << " loss: " << total_loss / seq_len << "\n";
        std::cout << generate(Wxh, Whh, Why, bh, by, idx_to_char, data[0], hidden_size, 100) << "\n\n";
    }

    if (epoch % 500 == 0 && epoch > 0) {
        save("data/Wxh.bin", Wxh);
        save("data/Whh.bin", Whh);
        save("data/Why.bin", Why);
        save("data/bh.bin",  bh);
        save("data/by.bin",  by);
        std::cout << "[saved]\n";
    }

    epoch++;
    if (epoch * seq_len + seq_len + 1 >= data.size()) epoch = 0;
}

    return 0;
}