#!/bin/bash
set -e
# Saved last 4 commands from history on ti. 20. mai 14:15:12 +0200 2025
cd ~/opm
bash tmp.sh
cd -
./../../../super_build_release/opm-simulators/bin/test_gpuflowproblem
