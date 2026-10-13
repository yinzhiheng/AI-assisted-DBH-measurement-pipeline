#pragma once

#include <filesystem>
#include <istream>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace batch
{
bool ReadCsvLine(std::istream& input, std::string& line);
std::vector<std::string> ParseCsvLine(const std::string& line);

using GroundTruthMap = std::unordered_map<std::string, std::optional<double>>;

// Reads normalized label CSV by uniqueName. Missing labels stay empty.
// ground_truth_diameter_m is explicitly in meters; source values/units are retained.
GroundTruthMap LoadGroundTruth(const std::filesystem::path& csvPath);
}
