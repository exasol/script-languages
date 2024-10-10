#ifndef SCRIPTOPTIONLINESCTPG_H
#define SCRIPTOPTIONLINESCTPG_H

#include <string>
#include <vector>
#include <map>
#include <ostream>

namespace ExecutionGraph
{

namespace OptionsLineParser
{

namespace CTPG
{

class ParserResult;

struct ScriptOption {
    std::string value;
    size_t idx_in_source;
    size_t size;

    bool operator==(const ScriptOption & right) const {
        return value == right.value && idx_in_source == right.idx_in_source && size == right.size;
    }
    /*
    Useful for gtest to print proper content when comparison fails.
    Copied the example from https://github.com/google/googletest/blob/main/docs/advanced.md#teaching-googletest-how-to-print-your-values
    */
    friend void PrintTo(const ScriptOption& option, std::ostream* os) {
        *os << "(" << option.value << "," << option.idx_in_source << "," << option.size << ")";
    }
};

using options_t = std::vector<ScriptOption>;

using options_map_t = std::map<std::string, options_t>;

/*!
 * \brief parseOptions Extracts syntactically valid options of form "%<option> <value>;" from UDF scripts.
 *
 * \param code Reference to string where the script code is stored.
 * \param result Result of all found options.
 * \param throwException Function to be called to throw exception.
 * \throws std::runtime_error if parsing fails
 */
void parseOptions(const std::string& code, options_map_t & result);

} //namespace CTPG

} // namespace OptionsLineParser

} // namespace ExecutionGraph

#endif // SCRIPTOPTIONLINESCTPG_H
