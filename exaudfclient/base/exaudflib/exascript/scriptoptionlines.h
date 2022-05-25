#ifndef SCRIPTOPTIONLINES_H
#define SCRIPTOPTIONLINES_H

#include <string>
#include <functional>

namespace ExecutionGraph
{

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
std::string extractOptionLine(std::string& code, const std::string option, const std::string whitespace, const std::string lineEnd, size_t& pos, std::function<void(const char*)> throwException);

} // namespace ExecutionGraph

#endif // SCRIPTOPTIONLINES_H
