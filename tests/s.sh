#!/bin/bash

cd -
bash tmp.sh
cd -
./../../../super_build_release/opm-simulators/bin/test_gpuflowproblem
