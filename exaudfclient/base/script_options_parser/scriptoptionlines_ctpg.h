#ifndef SCRIPTOPTIONLINESCTPG_H
#define SCRIPTOPTIONLINESCTPG_H

#include <string>
#include <functional>
#include <map>

namespace OptionsLineParser
{

struct OptionValue {
    std::string value;
    size_t idx;
};

class ParserResult {
    public:
        ParserResult() = default;
        using option_values_t = std::vector<OptionValue>;

        void get_option_values_and_remove_from_source_string(std::string & source, const std::string & key,
                                                             std::vector<std::string> result);

    private:
        void update_entries(const size_t start_idx, const size_t end_idx);

    private:
        using options_t = std::map<std::string, option_values_t>;
        options_t m_options;
};

/*!
 * \brief extractOptionLine Extracts syntactically valid option lines of form %<option> <values> [<values>] from UDF scripts.
 *
 * \param code Reference to string where the script code is stored.
 * \param option Option to be extracted. eg. "jvmoption", "jar", "env" etc.
 * \param whitespace String of characters that should be treated as white space characters.
 * \param lineEnd String of characters that should be treated as line end characters.
 * \param pos If option is found, contains the start position of the option. Otherwise, contains std::string::npos.
 * \param throwException Function to be called to throw exception.
 *
 * \return String with the option line.
 */
void parseOptions(const std::string& code, ParserResult & result, std::function<void(const char*)> throwException);

} // namespace OptionsLineParser

#endif // SCRIPTOPTIONLINESCTPG_H
