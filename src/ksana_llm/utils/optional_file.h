/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <Python.h>
#include <filesystem>

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/singleton.h"

namespace ksana_llm {

class OptionalFile {
 public:
  std::string& GetOptionalFile(const std::string& path_prefix, const std::string& path_name,
                               const std::string& file_name) {
    SearchOptionalFile(path_prefix, path_name, file_name);
    return target_file;
  }

 private:
  void SearchOptionalFile(const std::string& path_prefix, const std::string& path_name, const std::string& file_name) {
    // Search within the model path
    target_file = path_prefix + "/" + file_name;
    if (FileExists()) {
      return;
    }

    PyObject* sysPath = PySys_GetObject("path");
    if (sysPath && PyList_Check(sysPath)) {
      // Search within the ksana_llm packages
      PyObject* pathObj = PyList_GetItem(sysPath, 0);
      if (pathObj && PyUnicode_Check(pathObj) && PyUnicode_AsUTF8String(pathObj)) {
        std::string python_dir(PyBytes_AsString(PyUnicode_AsUTF8String(pathObj)));
        target_file = python_dir + "/" + path_name + "/" + file_name;
        if (FileExists()) {
          return;
        }
      }
      // Search within the python site-packages
      Py_ssize_t size = PyList_Size(sysPath);
      std::string package_suffix = "site-packages";
      for (Py_ssize_t i = 1; i < size; ++i) {
        PyObject* pathObj = PyList_GetItem(sysPath, i);
        if (pathObj && PyUnicode_Check(pathObj) && PyUnicode_AsUTF8String(pathObj)) {
          std::string python_dir(PyBytes_AsString(PyUnicode_AsUTF8String(pathObj)));
          if (python_dir.length() > package_suffix.length() &&
              python_dir.substr(python_dir.length() - package_suffix.length()) == package_suffix) {
            target_file = python_dir + "/ksana_llm/" + path_name + "/" + file_name;
            if (FileExists()) {
              return;
            }
          }
        }
      }
    }

    // Not Found
    target_file = "";
  }

  bool FileExists() {
    bool is_exists = std::filesystem::exists(target_file);
    KLLM_LOG_DEBUG << fmt::format("File {} is exists? {}", target_file, is_exists);
    return is_exists;
  }

  std::string target_file = "";
};

}  // namespace ksana_llm
