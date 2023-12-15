FROM mirrors.tencent.com/todacc/venus-std-base-cuda11.8:0.2.0

#MAINTAINER 维护者信息
LABEL MAINTAINER="karlluo"

USER root
# 安装必须软件
RUN GENERIC_REPO_URL="http://mirrors.tencent.com/repository/generic/venus_repo/image_res" \
    && cd /data/ \
    && wget -q $GENERIC_REPO_URL/gcc/gcc-11.2.0.zip \
    && unzip -q gcc-11.2.0.zip  \
    && cd gcc-releases-gcc-11.2.0 \
    && ./contrib/download_prerequisites \
    && ./configure --enable-bootstrap --enable-languages=c,c++ --enable-threads=posix --enable-checking=release --enable-multilib --with-system-zlib \
    && make --silent -j10 \
    && make --silent install \
    && gcc -v \
    && rm -rf /data/gcc-releases-gcc-11.2.0 /data/gcc-11.2.0.zip \
    && wget http://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2 \
    && bunzip2 gmp-6.1.0.tar.bz2 \
    && tar xvf gmp-6.1.0.tar \
    && cd gmp-6.1.0 \
    && ./configure \
    && make -j \
	&& make install -j \
    && cd .. \
    && rm -rf gmp-6.1.0 gmp-6.1.0.tar gmp-6.1.0.tar.bz2 \
	&& wget https://ftp.gnu.org/gnu/gdb/gdb-11.2.tar.gz \
    && tar vzxf gdb-11.2.tar.gz \
	&& cd gdb-11.2 \
    && ./configure \
	&& make -j \
	&& make install -j \
    && cd .. \
    && rm -rf gdb-11.2 gdb-11.2.tar.gz \
	&& wget https://ftp.gnu.org/gnu/bison/bison-3.0.4.tar.gz \
	&& tar vzxf bison-3.0.4.tar.gz \
	&& cd bison-3.0.4 \
	&& ./configure \
	&& make -j \
	&& make install -j \
    && cd .. \
    && rm -rf bison-3.0.4 bison-3.0.4.tar.gz \
    && wget https://ftp.gnu.org/gnu/binutils/binutils-2.41.tar.gz \
    && tar vzxf binutils-2.41.tar.gz \
    && cd binutils-2.41 \
    && ./configure --prefix=/usr \
    && make -j \
    && make install -j \
    && cd .. \
    && rm -rf binutils-2.41 binutils-2.41.tar.gz

RUN yum update -y \
    && yum install -y epel-release \
    && yum install -y centos-release-scl devtoolset-11 

RUN wget https://github.com/Kitware/CMake/releases/download/v3.28.0-rc5/cmake-3.28.0-rc5-linux-x86_64.sh \
	&& bash cmake-3.28.0-rc5-linux-x86_64.sh --prefix=/usr --skip-license \
    && git clone https://github.com/NVIDIA/nccl.git \
    && cd nccl \
    && git checkout v2.19.4-1 \
    && make --silent -j32 \
    && make install \
    && cd - \
    && rm -rf nccl \
    && git clone https://github.com/jemalloc/jemalloc.git \
    && cd jemalloc \
    && git checkout 5.3.0 \
    && bash autogen.sh \
    && ./configure \
    && make install -j \
    && rm -rf jemalloc

ENV LD_LIBRARY_PATH=/usr/local/lib64:/usr/local/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib64/:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:/usr/lib/nccl/:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH

RUN source ~/.bashrc \
    && wget -P /tmp http://mirrors.tencent.com/repository/generic/venus_repo/image_res/cpu/clean-layer.sh \
    && bash /tmp/clean-layer.sh

USER root    

RUN rm -rf /etc/alternatives/jre* \
	&& rm -rf /usr/lib/jvm/java* \
    && ln -sf /data/TencentKona-8.0.12-352/bin/java /usr/bin/java