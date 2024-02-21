/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "ksana_llm/utils/yaml_reader.h"
#include "ksana_llm/utils/ret_code.h"
#include "ksana_llm/utils/string_utils.h"
#include "yaml-cpp/node/node.h"

namespace ksana_llm {

Status YamlReader::LoadFile(const std::string& yaml_file) {
  // The try..cach is necessary, otherwise the program will cause exception when yaml file not exists.
  try {
    root_node_ = YAML::LoadFile(yaml_file);
  } catch (YAML::BadFile& e) {
    return Status(RET_INVALID_ARGUMENT, "YAML config " + yaml_file + " not exist.");
  }

  if (root_node_.Type() == YAML::NodeType::value::Undefined || root_node_.Type() == YAML::NodeType::value::Null ||
      root_node_.Type() == YAML::NodeType::value::Scalar) {
    return Status(RET_INVALID_ARGUMENT, "Load YAML config " + yaml_file + " error.");
  }

  return Status();
}

const YAML::Node YamlReader::GetRootNode() const {
  // Clone a full instance, otherwise the node will be changed after travel.
  return YAML::Clone(root_node_);
}

YAML::Node YamlReader::GetNodeFromDomain(YAML::Node root_node, const std::string& domain) {
  std::vector<std::string> vec = Str2Vector(domain, ".");

  YAML::Node node = root_node;
  for (const auto& s : vec) {
    if (!node[s]) {
      return YAML::Node();
    }
    node = node[s];
  }

  return node;
}

YAML::Node YamlReader::GetSequence(YAML::Node root_node, const std::string& domain) {
  YAML::Node node = GetNodeFromDomain(root_node, domain);
  if (node.IsSequence()) {
    return node;
  }

  return YAML::Node();
}

YAML::Node YamlReader::GetMap(YAML::Node root_node, const std::string& domain) {
  YAML::Node node = GetNodeFromDomain(root_node, domain);
  if (node.IsMap()) {
    return node;
  }

  return YAML::Node();
}

}  // namespace ksana_llm
