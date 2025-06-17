#!/bin/bash

# Simple test script to isolate profiling issue

EXECUTABLE="./Build/conv_benchmark"
PROFILE_SCRIPT="./tools/profile.sh"

echo "Testing profiling call..."
echo "Command: bash \"$PROFILE_SCRIPT\" --executable \"$EXECUTABLE\" --mode summary --size 256 --kernel 3 --tool nsys"

# Try the call
bash "$PROFILE_SCRIPT" --executable "$EXECUTABLE" --mode summary --size 256 --kernel 3 --tool nsys
