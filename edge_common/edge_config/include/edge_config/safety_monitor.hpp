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

struct SafetyRule {
    std::string name;   // rule name (yaml key)
    int index;          // errorState index
    double limit;       // limit after unit conversion
    std::string unit;   // unit used in logging
};

class SafetyMonitor
{
public:
    explicit SafetyMonitor(const std::string& yaml_path="");

    bool evaluateSafety(const std::vector<double>& errorState, double obstacle_distance, double goal_distance);

private:
    std::vector<SafetyRule> rules_;
    bool is_curr_safe_ = true;
    bool is_last_safe_ = true;
    std::vector<std::string> violation_reasons_;

    void loadFromYaml(const std::string& yaml_path);
    std::string parseYamlPath(const std::string& user_path);
    void safetyStatusLogging();

    double convertLimit(double raw, const std::string& unit);
    void printLoadedRules() const;
};

}  // namespace edge::config
