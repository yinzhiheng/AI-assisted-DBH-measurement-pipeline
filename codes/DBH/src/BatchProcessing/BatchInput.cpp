#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "BatchInput.h"
#include "load.h"
#include <fstream>
#include <locale>

namespace fs = std::filesystem;

namespace batch
{
bool ReadCsvLine(std::istream& input, std::string& line)
{
    line.clear();
    char c;
    while (input.get(c)) {
        if (c == '\r') {
            if (input.peek() == '\n') input.get();
            return true;
        }
        if (c == '\n') return true;
        line += c;
    }
    return !line.empty();
}

std::vector<std::string> ParseCsvLine(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                field += '"';
                ++i;
            }
            else quoted = !quoted;
        }
        else if (c == ',' && !quoted) {
            fields.push_back(Trim(field));
            field.clear();
        }
        else field += c;
    }
    if (quoted) throw std::runtime_error("Unclosed quoted CSV field");
    fields.push_back(Trim(field));
    return fields;
}

GroundTruthMap LoadGroundTruth(const fs::path& csvPath)
{
    std::ifstream input(csvPath, std::ios::binary);
    std::string line;
    if (!input || !ReadCsvLine(input, line)) throw std::runtime_error("Cannot read label CSV");
    if (line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
    const std::vector<std::string> expected{
        "uniqueName", "ground_truth_source_value", "ground_truth_source_unit", "ground_truth_diameter_m"};
    if (ParseCsvLine(line) != expected) throw std::runtime_error("Unexpected label CSV columns");
    GroundTruthMap labels;
    while (ReadCsvLine(input, line)) {
        if (Trim(line).empty()) continue;
        const auto fields = ParseCsvLine(line);
        if (fields.size() != expected.size() || fields[0].empty())
            throw std::runtime_error("Invalid label CSV record");
        std::optional<double> diameterM;
        if (!fields[3].empty()) {
            std::istringstream number(fields[3]);
            number.imbue(std::locale::classic());
            double meters = 0;
            if (!(number >> meters) || !(number >> std::ws).eof() || !std::isfinite(meters) || meters <= 0)
                throw std::runtime_error("Invalid ground-truth diameter: " + fields[0]);
            diameterM = meters;
        }
        if (!labels.emplace(fields[0], diameterM).second)
            throw std::runtime_error("Duplicate label: " + fields[0]);
    }
    if (input.bad()) throw std::runtime_error("Error reading label CSV");
    return labels;
}
}
