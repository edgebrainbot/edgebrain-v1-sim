#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>

/*******   ANSI colors for printing   *******/ 
#define C_RESET   "\033[0m"
#define C_TITLE   "\033[1;36m"   // Cyan
#define C_HEADER  "\033[1;37m"   // White bold
#define C_VALUE   "\033[0;33m"   // Yellow
#define C_UNIT    "\033[0;32m"   // Green


namespace edge::config
{

// Rule type: 
// 1. relative = |x - x_ref| <= limit, 
// 2. absolute = lower < x < upper
enum class RuleType { RELATIVE, ABSOLUTE };

struct SafetyRule {
    RuleType type;      // RELATIVE or ABSOLUTE
    std::string name;   // rule name (yaml key)
    int index;          // state index
    double limit;       // limit for relative mode (after unit conversion)
    double lower;       // lower bound for absolute mode
    double upper;       // upper bound for absolute mode
    std::string unit;   // unit used in logging
};

class SafetyMonitor
{
public:
    explicit SafetyMonitor(const std::string& yaml_path="");

    // type: RELATIVE (state=errorState) or ABSOLUTE (state=robotState)
    bool evaluateSafety(RuleType ruleType,
                        const std::vector<double>& state,
                        double obstacle_distance = 0.0, 
                        double goal_distance = 0.0);

private:
    std::vector<SafetyRule> rules_;
    bool is_curr_safe_ = true;
    bool is_last_safe_ = true;
    std::vector<std::string> violation_reasons_;

    void loadFromYaml(const std::string& yaml_path);
    std::string parseYamlPath(const std::string& user_path);
    void safetyStatusLogging();

    double convertValue(double raw, const std::string& unit);
    void printLoadedRules() const;
};

}  // namespace edge::config
