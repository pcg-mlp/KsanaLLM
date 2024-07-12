/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include "ksana_llm/utils/logger.h"
#include "ksana_llm/utils/status.h"
#include "yaml-cpp/yaml.h"

namespace ksana_llm {

// A reader used to parse YAML config file.
class YamlReader {
 public:
  // Load yaml file from disk.
  Status LoadFile(const std::string& yaml_file);

  // Get the root node of yaml.
  const YAML::Node GetRootNode() const;

  // Get a scalar value from yaml, return default value if not exists.
  template <typename T>
  T GetScalar(YAML::Node root_node, const std::string& domain, const T& default_val = T()) {
    YAML::Node node = GetNodeFromDomain(root_node, domain);
    if (node.IsScalar()) {
      return node.as<T>();
    }

    return default_val;
  }

  // Get a sequeue node from yaml.
  YAML::Node GetSequence(YAML::Node root_node, const std::string& domain);

  // Get a map node from yaml.
  YAML::Node GetMap(YAML::Node root_node, const std::string& domain);

 private:
  // Get yaml node from damin.
  YAML::Node GetNodeFromDomain(YAML::Node root_node, const std::string& domain);

 private:
  // The yaml node config.
  YAML::Node root_node_;
};

}  // namespace ksana_llm
