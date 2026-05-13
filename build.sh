#!/bin/bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
cmake -S . -B build -G "Ninja" 2>/dev/null || cmake -S . -B build -G "Ninja"
cmake --build build && ./build/rnn.exe
