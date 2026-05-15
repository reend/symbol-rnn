# RNN symbol predictor

A from-scratch character-level Recurrent Neural Network trained on Shakespeare's plays. No ML frameworks — only standard C++ and OpenBLAS for BLAS-accelerated matrix operations.

## What it does

The network reads `data/input.txt` character by character, learns to predict the next character, and eventually generates Shakespeare-like text.

**Architecture:**

- Input layer: one-hot encoded character (vocab size × 1)
- Hidden layer: 128 units with tanh activation
- Output layer: softmax over the vocabulary
- Training: BPTT (Backpropagation Through Time) over sequences of 25 characters
- Optimizer: SGD with gradient clipping (±5) and learning rate 0.01

**Weights** (`Wxh`, `Whh`, `Why`, `bh`, `by`) are saved to `data/*.bin` every 500 epochs and loaded on startup to resume training.

Every 100 epochs the current loss and a 100-character sample are printed to stdout.

## Dependencies

| Dependency | Version |
|------------|---------|
| C++ compiler | C++17 |
| CMake | ≥ 3.14 |
| Ninja | any |
| OpenBLAS | via MSYS2 UCRT64 |

Install OpenBLAS on Windows via MSYS2:

```bash
pacman -S mingw-w64-ucrt-x86_64-openblas
```

## Build

**Windows (cmd):**

```bat
build.bat
```

**Windows (Git Bash / MSYS2):**

```bash
./build.sh
```

**Manual:**

```bash
cmake -S . -B build -G "Ninja" \
  -DCMAKE_CXX_COMPILER=C:/msys64/ucrt64/bin/g++.exe \
  -DCMAKE_C_COMPILER=C:/msys64/ucrt64/bin/gcc.exe \
  -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64/bin/ninja.exe
cmake --build build
```

The binary is placed at `build/rnn.exe`.

## Run

```bash
./build/rnn.exe
```

Make sure `data/input.txt` exists before running. The training loop runs indefinitely, wrapping back to the start of the text when the end is reached.

## Project structure

```
nn/
├── src/main.cpp      # entire implementation
├── data/
│   ├── input.txt     # training corpus (Shakespeare)
│   └── *.bin         # saved weights (generated at runtime)
├── CMakeLists.txt
├── build.bat         # Windows build script
└── build.sh          # MSYS2/Bash build script
```
