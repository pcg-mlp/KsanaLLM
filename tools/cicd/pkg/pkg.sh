#!/usr/bin/env bash

export KSANA_PROJ_BASE=$(pwd)/KsanaLLM
tar cvf $KSANA_PROJ_BASE/build/package.tar                                    \
        $KSANA_PROJ_BASE/csrc/                                                \
        $KSANA_PROJ_BASE/tools/                                               \
        $KSANA_PROJ_BASE/python/                                              \
        $KSANA_PROJ_BASE/3rdparty/                                            \
        $KSANA_PROJ_BASE/examples/                                            \
        $KSANA_PROJ_BASE/benchmarks/                                          \
        $KSANA_PROJ_BASE/build/lib/*.so*                                      \
        $KSANA_PROJ_BASE/build/tbb_install/lib64/libtbb.so*                   \
        $KSANA_PROJ_BASE/build/third_party/DeepGEMM/                          \
        ~/.cache/KsanaLLM
