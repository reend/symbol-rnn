#!/bin/bash
export PATH="/c/msys64/ucrt64/bin:$PATH"
cmake -S . -B build -G "Ninja"
cmake --build build
