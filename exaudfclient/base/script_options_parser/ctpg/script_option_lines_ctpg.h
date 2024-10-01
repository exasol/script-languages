#ifndef SCRIPTOPTIONLINESCTPG_H
#define SCRIPTOPTIONLINESCTPG_H

#include <string>
#include <functional>
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
    friend void PrintTo(const ScriptOption& option, std::ostream* os) {
        *os << "(" << option.value << "," << option.idx_in_source << "," << option.size << ")";
    }
};

using options_t = std::vector<ScriptOption>;

using options_map_t = std::map<std::string, options_t>;

/*!
 * \brief extractOptionLine Extracts syntactically valid option lines of form %<option> <values> [<values>] from UDF scripts.
 *
 * \param code Reference to string where the script code is stored.
 * \param result Result of all found options.
 * \param throwException Function to be called to throw exception.
 *
 */
void parseOptions(const std::string& code, options_map_t & result, std::function<void(const char*)> throwException);

} //namespace CTPG

} // namespace OptionsLineParser

} // namespace ExecutionGraph

#endif // SCRIPTOPTIONLINESCTPG_H
