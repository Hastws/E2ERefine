#ifndef INCLUDE_AUTOALG_FUNCTIONS_H
#define INCLUDE_AUTOALG_FUNCTIONS_H

#include "discrete_activation.h"
#include "sigmoid.h"
#include "softmax.h"

namespace autoalg {
    using Function = std::function<Real(Real)>;

    inline std::string ToLower(const std::string &s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(), ::tolower);
        return result;
    }

    class FunctionsManager {
    public:
        static FunctionsManager &Instance() {
            static FunctionsManager instance;
            return instance;
        }

        void RegisterFunction(const std::string &name, const Function &func) {
            const std::string key = ToLower(name);
            if (function_map_.count(key)) return; // Avoid duplicate registration

            const int id = static_cast<int>(id_to_name_.size());
            function_map_[key] = func;
            name_to_id_[key] = id;
            id_to_name_.push_back(key); // all lowercase

            cache_valid_ = false;
        }

        Real Call(const std::string &name, const Real &input) const {
            const auto it = function_map_.find(ToLower(name));
            if (it != function_map_.end()) {
                return it->second(input);
            }
            throw std::runtime_error("Function not found: " + name);
        }

        std::vector<std::string> GetAllFunctionNames() const {
            std::vector<std::string> names;
            for (const auto &pair: function_map_) {
                names.push_back(pair.first);
            }
            return names;
        }

        std::size_t GetNumberOfFunctions() const {
            return function_map_.size();
        }

        int GetFunctionId(const std::string &name) const {
            const auto it = name_to_id_.find(ToLower(name));
            if (it != name_to_id_.end()) return it->second;
            throw std::runtime_error("Function ID not found for: " + name);
        }

        std::string GetFunctionName(const int id) const {
            if (id >= 0 && id < static_cast<int>(id_to_name_.size()))
                return id_to_name_[id];
            throw std::runtime_error("Invalid function ID: " + std::to_string(id));
        }

        const char *GetZeroSeparatedFunctionNames() {
            if (!cache_valid_) {
                function_names_zero_separated_.clear();
                for (const auto &name: id_to_name_) {
                    function_names_zero_separated_.append(name);
                    function_names_zero_separated_.push_back('\0');
                }
                cache_valid_ = true;
            }
            return function_names_zero_separated_.c_str();
        }

    private:
        bool cache_valid_ = false;
        std::string function_names_zero_separated_;
        std::unordered_map<std::string, Function> function_map_;
        std::unordered_map<std::string, int> name_to_id_; // name → id
        std::vector<std::string> id_to_name_; // id → name
        FunctionsManager() = default;
    };
}

#define REGISTER_FUNCTION(name, func) \
static bool Registered_##name = []() { \
::autoalg::FunctionsManager::Instance().RegisterFunction(#name, func); \
return true; \
}()

#endif // INCLUDE_AUTOALG_FUNCTIONS_H
