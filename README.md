# Usage

Start docker container with following image:
```
mirrors.tencent.com/todacc/venus-numerous-llm:0.1.5
```

## Compile
```bash
mkdir build
cd build
cmake ..
make -j
```

## Run

```bash
# run unittests
./bin/numerous_llm_unittests
# run standalone demo
./bin/numerous_llm
# check runing status
# open another terminal or session
cat ./log/numerous_llm.log
```

## Code format command line

please format your code before submit a merge request

```bash
# prepare clang-format
# pip install clang-format
# prefer clang-format 17.0.5
cd ${GIT_PROJECT_ROOT_DIR}
clang-for
clang-format -i ${CODE_YOUR_EDIT}
```