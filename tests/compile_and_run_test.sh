#!/bin/bash
set -e
# Define the list of executables to compile and run
# execs=("test_flow_simple_gpu")
# execs=("test_flow_simple")
execs=("test_gpuflowproblem")

# Check if argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 [d|r]"
    echo "  d: Debug mode"
    echo "  r: Release mode"
    exit 1
fi

# Set build type based on argument
if [ "$1" = "d" ]; then
    build_type="debug"
    build_dir="super_build_debug"
elif [ "$1" = "r" ]; then
    build_type="release"
    build_dir="super_build_release"
else
    echo "Invalid argument: $1"
    echo "Usage: $0 [d|r]"
    echo "  d: Debug mode"
    echo "  r: Release mode"
    exit 1
fi

# Navigate to the root directory
cd ../../..

# Compile all executables and save logs
for exec in "${execs[@]}"; do
    echo "Compiling $exec in $build_type mode..."
    bash makesuperbuild.sh -HC $build_type $exec | tee ~/error_${exec}_${build_type}.txt
done

# Navigate back to the original directory
cd -

# Run all executables
for exec in "${execs[@]}"; do
    echo "Running $exec from $build_type build..."
    ./../../../${build_dir}/opm-simulators/bin/$exec very_simple_deck.DATA
done
