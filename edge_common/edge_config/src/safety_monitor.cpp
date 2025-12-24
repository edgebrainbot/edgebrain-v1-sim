#include <cmath>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include "edge_config/safety_monitor.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>


// ANSI colors
static const std::string CYAN  = "\033[36m";
static const std::string YELLOW = "\033[33m";
static const std::string GREEN = "\033[32m";
static const std::string RESET = "\033[0m";

static constexpr bool USE_COLOR = true;

static std::string color(const std::string& s, const std::string& c) {
    return USE_COLOR ? (c + s + RESET) : s;
}

// Helper: print centered text inside a width
static std::string center(const std::string& text, size_t width) {
    if (text.size() >= width) return text;
    size_t total = width - text.size();
    size_t left  = total / 2;
    size_t right = total - left;
    return std::string(left, ' ') + text + std::string(right, ' ');
}

namespace edge::config
{

SafetyMonitor::SafetyMonitor(const std::string& yaml_path)
{
    std::string final_path = parseYamlPath(yaml_path);
    loadFromYaml(final_path);
    printLoadedRules();
}

/**
 * Determine the YAML path to use
 */
std::string SafetyMonitor::parseYamlPath(const std::string& user_path)
{
    if (!user_path.empty()) {
        return user_path; // user override
    }

    // default YAML path
    std::string pkg_share = ament_index_cpp::get_package_share_directory("edge_config");
    return pkg_share + "/config/safety_rules.yaml";
}

/**
 * Load safety rules from YAML file
 */
void SafetyMonitor::loadFromYaml(const std::string& yaml_path)
{
    YAML::Node root = YAML::LoadFile(yaml_path);

    if (!root["safety_rules"]) {
        throw std::runtime_error("YAML missing 'safety_rules' section");
    }

    YAML::Node rules_node = root["safety_rules"];
    rules_.clear();

    for (auto it = rules_node.begin(); it != rules_node.end(); ++it) {
        std::string name = it->first.as<std::string>();
        YAML::Node node = it->second;

        int index = node["index"].as<int>();
        std::string unit = node["unit"].as<std::string>();
        
        // Determine rule type (default: relative)
        std::string type_str = node["type"] ? node["type"].as<std::string>() : "relative";
        RuleType type = (type_str == "absolute") ? RuleType::ABSOLUTE : RuleType::RELATIVE;
        
        SafetyRule rule;
        rule.name = name;
        rule.index = index;
        rule.unit = unit;
        rule.type = type;
        rule.limit = 0.0;
        rule.lower = 0.0;
        rule.upper = 0.0;
        
        if (type == RuleType::RELATIVE) {
            // Relative mode: |x| <= limit
            double raw_limit = node["limit"].as<double>();
            rule.limit = convertValue(raw_limit, unit);
        } else if (type == RuleType::ABSOLUTE) {
            // Absolute mode: lower < x < upper
            if (!node["bound"] || !node["bound"].IsSequence() || node["bound"].size() != 2) {
                throw std::runtime_error("Absolute rule '" + name + "' requires 'bound: [lower, upper]'");
            }
            double raw_lower = node["bound"][0].as<double>();
            double raw_upper = node["bound"][1].as<double>();
            rule.lower = convertValue(raw_lower, unit);
            rule.upper = convertValue(raw_upper, unit);
        } else {
            throw std::runtime_error("Unknown rule type for rule '" + name + "'");
        }
        
        rules_.push_back(rule);
    }
}

/**
 * Convert the user-provided value to internal value (always SI)
 */
double SafetyMonitor::convertValue(double raw, const std::string& unit)
{
    if (unit == "deg") {
        return raw * M_PI / 180.0; // convert to rad
    }
    if (unit == "rad") {
        return raw;
    }
    if (unit == "m") {
        return raw;
    }
    if (unit == "m/s") {
        return raw;
    }
    if (unit == "rad/s") {
        return raw;
    }
    std::cerr << "⚠️ Unknown unit: " << unit << " (treat as raw SI)" << std::endl;
    return raw; 
}

bool SafetyMonitor::evaluateSafety(RuleType ruleType,
                                   const std::vector<double>& state,
                                   double obstacle_distance,
                                   double goal_distance)
{
    if (state.size() < 12) {
        throw std::runtime_error("State vector too small in evaluateSafety()");
    }

    violation_reasons_.clear();
    bool is_safe = true;

    for (const auto& r : rules_) {
        // Only check rules matching the specified type
        if (r.type != ruleType) {
            continue;
        }

        if (r.index == -1) {
            // Special case: obstacle distance
            if (r.name == "obstacle_distance" && obstacle_distance < r.limit) {
                is_safe = false;
                violation_reasons_.push_back(
                    r.name + ": " + std::to_string(obstacle_distance) + " < " + 
                    std::to_string(r.limit) + " (" + r.unit + ")"
                );
            }
            // Special case: goal distance
            if (r.name == "goal_distance" && goal_distance < r.limit) {
                is_safe = false;
                violation_reasons_.push_back(
                    r.name + ": " + std::to_string(goal_distance) + " < " + 
                    std::to_string(r.limit) + " (" + r.unit + ")"
                );
            }
            continue;
        }

        double v = state[r.index];
        
        if (r.type == RuleType::RELATIVE) {
            // Relative mode: |error_x| <= limit
            // state is errorState (already contains error/deviation)
            if (std::abs(v) > r.limit) {
                is_safe = false;
                violation_reasons_.push_back(
                    r.name + ": |" + std::to_string(v) + "| = " + 
                    std::to_string(std::abs(v)) + " > " + 
                    std::to_string(r.limit) + " (" + r.unit + ")"
                );
            }
        } else if (r.type == RuleType::ABSOLUTE) {
            // Absolute mode: lower < x < upper
            // state is robotState
            if (v <= r.lower || v >= r.upper) {
                is_safe = false;
                violation_reasons_.push_back(
                    r.name + ": " + std::to_string(v) + " not in (" + 
                    std::to_string(r.lower) + ", " + std::to_string(r.upper) + ") (" + r.unit + ")"
                );
            }
        } else {
            throw std::runtime_error("Unknown rule type in evaluateSafety()");
        }
    }

    is_last_safe_ = is_curr_safe_;
    is_curr_safe_ = is_safe;

    safetyStatusLogging();
    return is_safe;
}

// --------------------------------------------------------
void SafetyMonitor::safetyStatusLogging()
{
    // Logging
    if (!is_curr_safe_) {
        std::string msg = "⚠️ Current System is UNSAFE! Reasons: ";
        for (auto &r : violation_reasons_) msg += r + " | ";

        if (is_last_safe_)
            msg += "System leaving the learning space! Safety Triggered.";
        else
            msg += "System still outside the learning space.";

        std::cerr << msg << std::endl;
    } else {
        if (!is_last_safe_) {
        std::cout << "✅ System returned to the learning space." << std::endl;
        } else {
        std::cout << "✅ System remains within the learning space." << std::endl;
        }
    }
}


/**
 * Print loaded safety rules in a formatted table
 */
void SafetyMonitor::printLoadedRules() const
{
    const int NAME_W  = 10;
    const int IDX_W   = 6;
    const int TYPE_W  = 10;
    const int LIMIT_W = 22;
    const int UNIT_W  = 8;

    int total_w = NAME_W + IDX_W + TYPE_W + LIMIT_W + UNIT_W + 12;

    auto line = [&]() {
        std::cout << "  " << std::string(total_w, '-') << "\n";
    };

    // ------------------------------------------------------------
    // Unified formatter factory
    // makeFormatter(width).text(str)
    // makeFormatter(width).number(double)
    // ------------------------------------------------------------
    struct Formatter {
        int width;

        std::string text(const std::string &s) const {
        int pad = width - (int)s.size();
        if (pad <= 0) return s;
        int left = pad / 2;
        int right = pad - left;
        return std::string(left, ' ') + s + std::string(right, ' ');
        }

        std::string number(double v, int precision = 6) const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(precision) << v;
        return text(ss.str());
        }
    };

    auto makeFormatter = [&](int w) { return Formatter{w}; };

    auto FName  = makeFormatter(NAME_W);
    auto FIdx   = makeFormatter(IDX_W);
    auto FType  = makeFormatter(TYPE_W);
    auto FLimit = makeFormatter(LIMIT_W);
    auto FUnit  = makeFormatter(UNIT_W);

    // ------------------------------------------------------------
    // Title
    // ------------------------------------------------------------
    std::string title = "----------  🚨 Safety Rules Loaded  ----------";
    int pad = (total_w - (int)title.size()) / 2;
    std::cout << "\n  " << C_TITLE
              << std::string(pad, ' ') << title
              << C_RESET << "\n";

    // ------------------------------------------------------------
    // Header
    // ------------------------------------------------------------
    std::cout << "  "
              << C_HEADER << FName.text("Name")  << C_RESET << " | "
              << C_HEADER << FIdx.text("Idx")    << C_RESET << " | "
              << C_HEADER << FType.text("Type")  << C_RESET << " | "
              << C_HEADER << FLimit.text("Limit/Bounds")<< C_RESET << " | "
              << C_HEADER << FUnit.text("Unit")  << C_RESET
              << "\n";

    line();

    // ------------------------------------------------------------
    // Rows
    // ------------------------------------------------------------
    for (const auto& r : rules_) {
        std::string bound_str;
        std::string type_str;
        if (r.type == RuleType::RELATIVE) {
            type_str = "relative";
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << r.limit;
            bound_str = "±" + ss.str();
        } else {
            type_str = "absolute";
            std::ostringstream ss;
            ss << std::fixed << std::setprecision(4) << "(" << r.lower << ", " << r.upper << ")";
            bound_str = ss.str();
        }
        
        std::cout << "  "
                  << FName.text(r.name) << " | "
                  << FIdx.text(std::to_string(r.index)) << " | "
                  << FType.text(type_str) << " | "
                  << C_VALUE << FLimit.text(bound_str) << C_RESET << " | "
                  << C_UNIT  << FUnit.text(r.unit)     << C_RESET
                  << "\n";
    }

    line();
}

}  // namespace edge::config
