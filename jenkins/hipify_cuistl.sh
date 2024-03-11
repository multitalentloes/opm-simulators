#!/bin/bash

function hipify_recursively() {
    target_dir=$1
    find . -regex '\(.*\.cpp$\)\|\(.*\.hpp$\)\|\(.*\.cu$\)' -type f -exec bash -c 'hipify-perl "$1" > "$2"' _ {} "$target_dir/{}" \;
    # expand includes so we only need include_directories (path to hip)
    cd $target_dir
    sed -i 's/^#include <hipblas\.h>/#include <hipblas\/hipblas.h>/g' $(find . -type f)
    sed -i 's/^#include <hipsparse\.h>/#include <hipsparse\/hipsparse.h>/g' $(find . -type f)
    sed -i 's/cuistl\//hipistl\//g' $(find . -type f)
}

# the script is inteded to be run like this: bash hipify_cuistl.sh ${CMAKE_BUILD_DIR} ${CMAKE_BINARY_DIR}
opm_src=$1
build_path=$2

# get path to .../lingalg/hipistl
linalg_src=opm/simulators/linalg
hipistl_lingalg_src=$build_path/$linalg_src/hipistl
hipistl_tests_src=$build_path/tests/hipistl

# make directory where we place linalg/hipistl and tests/hipistl
mkdir -p $hipistl_lingalg_src/detail
mkdir -p $hipistl_tests_src

# hipify linalg
cd $opm_src/$linalg_src/cuistl
hipify_recursively $hipistl_lingalg_src

# hipify tests
cd $opm_src/tests/cuistl
hipify_recursively $hipistl_tests_src
