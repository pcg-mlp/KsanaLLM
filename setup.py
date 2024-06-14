# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import pathlib
import shutil

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as build_ext_orig
from distutils.file_util import copy_file


class CMakeExtension(Extension):

    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


def is_run_on_npu_device() -> bool:
    try:
        import torch_npu
        return True
    except:
        return False


class build_ext(build_ext_orig):

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        # these dirs will be created in build_py, so if you don't have
        # any python sources to bundle, the dirs will be missing
        build_temp = pathlib.Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        if not extdir.exists():
            extdir.mkdir(parents=True, exist_ok=True)

        # example of cmake args
        config = 'Debug' if self.debug else 'Release'
        cmake_args = []
        if is_run_on_npu_device():
            cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
                str(extdir.parent.absolute()),
                '-DCMAKE_BUILD_TYPE=' + config,
                '-DWITH_TESTING=OFF',
                '-DWITH_CUDA=OFF',
                '-DWITH_ACL=ON',
            ]
        else:
            cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' +
                str(extdir.parent.absolute()), '-DCMAKE_BUILD_TYPE=' + config,
                '-DWITH_TESTING=OFF',
                '-DSM=80,86,89'
            ]

        # example of build args
        build_args = ['--config', config, '--', '-j']

        os.chdir(str(build_temp))
        self.spawn(['cmake', str(cwd)] + cmake_args)
        if not self.dry_run:
            self.spawn(['cmake', '--build', '.'] + build_args)
        # Troubleshooting: if fail on line above then delete all possible
        # temporary CMake files including "CMakeCache.txt" in top level dir.
        os.chdir(str(cwd))
        # cp all temp dir's lib to cwd for dist package
        deps_lib = cwd.joinpath('src/ksana_llm/python/lib')
        build_temp_lib = build_temp.joinpath('lib')
        deps_lib.mkdir(parents=True, exist_ok=True)
        target_libs = ["libtorch_serving.so", "libloguru.so"]
        for target_lib in target_libs:
            # for local develop
            copy_file(str(build_temp_lib.joinpath(target_lib)), str(deps_lib))
            # for wheel pacakge
            copy_file(str(build_temp_lib.joinpath(target_lib)),
                      str(extdir.parent.absolute()))
        # copy optional weight map to cwd for wheel package
        def ignore_files(dir, files):
            return [file for file in files if not file.endswith('.json')]

        need_dirs = [
            'weight_map', 
            'ksana_plugin'
        ]
        for need_dir in need_dirs:
            src_dir = os.path.join('src/ksana_llm/python', need_dir)
            dst_dir = os.path.join(extdir.parent.absolute(), os.path.join("ksana_llm", need_dir))
            shutil.copytree(src_dir,
                            dst_dir,
                            dirs_exist_ok=True,
                            ignore=ignore_files)


setup(name='ksana_llm',
      version='v0.0.15',
      author='ksana_llm',
      author_email='ksana_llm@tencent.com',
      description='Ksana LLM inference server',
      platforms='python3',
      url='https://xxx/KsanaLLM/',
      packages=find_packages('src/ksana_llm/python'),
      package_dir={
          '': 'src/ksana_llm/python',
      },
      ext_modules=[CMakeExtension('ksana_llm')],
      python_requires='>=3',
      cmdclass={
          'build_ext': build_ext,
      })
