#!/usr/bin/env bash

export KSANA_PROJ_BASE=$(pwd)/KsanaLLM
rm -rf $KSANA_PROJ_BASE
mkdir -p $KSANA_PROJ_BASE
rm -rf ~/.cache/KsanaLLM/*
tar -C / -xvf package.tar

cd $KSANA_PROJ_BASE/build/third_party/DeepGEMM
pip install .