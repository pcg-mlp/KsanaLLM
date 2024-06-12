#include <vector>
#include <string>


namespace ksana_llm {

std::vector<std::string> SearchLocalPath(const std::string& model_path, bool& is_safetensors);

} // namespace ksana_llm