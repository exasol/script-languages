#include "scriptoptionlines.h"
#include <string>
#include <sstream>



namespace ExecutionGraph
{

std::string extractOptionLine(std::string& code, const std::string option, const std::string whitespace, const std::string lineEnd, size_t& pos, std::function<void(const char*)>throwException)
{
    std::string result;
    size_t startPos = code.find(option);
    bool ignore_option = false;

    if (startPos != std::string::npos)
    {
        // If there is anything other than whitespaces preceeding the option line, the option must be ignored.
        // Look for line end character in the previous line.
        size_t column_0 = code.find_last_of("\r\n", startPos);
        // If line end character was not found, then the option must be on the first line in the script.
        if (column_0 == std::string::npos)
        {
            // Look for first non white space character in the code.
            size_t firstNonWhiteSpace = code.find_first_not_of(whitespace);
            if (firstNonWhiteSpace == startPos)
            {
                // Set flag to NOT ignore the option.
                ignore_option = false;
            }
            else
            {
                // Set flag to ignore the option.
                ignore_option = true;
            }
        }
        else
        {
            // column_0 is pointing at line end character in previous line.
            // Increment it to point to first character of the line with option.
            column_0++;
            // Look for first character that is not whitespace
            size_t firstNonWhiteSpaceInLine = code.find_first_not_of(whitespace, column_0);
            if (firstNonWhiteSpaceInLine == startPos)
            {
                // Set flag to NOT ignore the option.
                ignore_option = false;
            }
            else
            {
                // Set flag to ignore the option.
                ignore_option = true;
            }
        }

        if (!ignore_option)
        {
            // Process the option.
            // Find the first value for the option.
            size_t firstPos = startPos + option.length();
            firstPos = code.find_first_not_of(whitespace, firstPos);
            if (firstPos == std::string::npos) {
                std::stringstream ss;
                ss << "No values found for " << option << " statement";
                throwException(ss.str().c_str());
            }
            // Find the end of line.
            size_t lastPos = code.find_first_of(lineEnd + "\r\n", firstPos);
            if (lastPos == std::string::npos || code.compare(lastPos, lineEnd.length(), lineEnd) != 0) {
                std::stringstream ss;
                ss << "End of " << option << " statement not found";
                throwException(ss.str().c_str());
            }
            // If no values were found
            if (firstPos >= lastPos) {
                std::stringstream ss;
                ss << "No values found for " << option << " statement";
                throwException(ss.str().c_str());
            }
            // If no values were found
            size_t optionsEnd = code.find_last_not_of(whitespace, lastPos - 1);
            if (optionsEnd == std::string::npos || optionsEnd < firstPos) {
                std::stringstream ss;
                ss << "No values found for " << option << " statement";
                throwException(ss.str().c_str());
            }
            result = code.substr(firstPos, optionsEnd - firstPos + 1);
            code.erase(startPos, lastPos - startPos + 1);
            pos = startPos;
        }
        else
        {
            // Ignore the option.
            pos = std::string::npos;
        }
    }
    else
    {
        // Option was not found.
        pos = std::string::npos;
    }
    return result;
}


} // namespace ExecutionGraph
