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
