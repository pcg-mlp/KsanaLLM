/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/singleton.h"

#include "logger.h"
#include <Python.h>

namespace ksana_llm {

class OptionalWeightMap {
  public:
    static std::shared_ptr<OptionalWeightMap> GetInstance() {
      return Singleton<OptionalWeightMap>::GetInstance();;
    }

    // TODO(zezhao): 初始化时,加载pip目录以及本地开发目录所有 weight_map.json;
    //               对于每个模型实例, 追加读取模型目录下的 weight_map.json
    std::string& GetOptionalWeightMap(std::string& model_path, std::string& model_type, bool force_reload = false) {
      if (force_reload || !is_loaded) {
        SearchWeightMap(model_path, model_type);
      }
      is_loaded = true;
      return optional_weight_map_;
    }

  private:
    // Search for the optional_weight_map.json file
    void SearchWeightMap(std::string& model_path, std::string& model_type) {
      // Search within the model path
      optional_weight_map_ = model_path + "/" + model_type + "_weight_map.json";
      if (FileExists()) {
        return;
      }

      PyObject* sysPath = PySys_GetObject("path");
      if (sysPath && PyList_Check(sysPath)) {
        // Search within the ksana-llm packages
        PyObject* pathObj = PyList_GetItem(sysPath, 0);
        if (pathObj && PyUnicode_Check(pathObj) && PyUnicode_AsUTF8String(pathObj)) {
          std::string python_dir(PyBytes_AsString(PyUnicode_AsUTF8String(pathObj)));
          optional_weight_map_ = python_dir + "/weight_map/" + model_type + "_weight_map.json";
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
              optional_weight_map_ = python_dir + "/ksana-llm/weight_map/" + model_type + "_weight_map.json";
              if (FileExists()) {
                return;
              }
            }
          }
        }
      }

      // Not Found
      optional_weight_map_ = "";
    }

    bool FileExists() {
      bool is_exists = std::filesystem::exists(optional_weight_map_);
      NLLM_LOG_DEBUG << fmt::format("File {} is exists? {}", optional_weight_map_, is_exists);
      return is_exists;
    }

    std::string optional_weight_map_ = "";
    bool is_loaded = false;
};

}  // namespace ksana_llm
