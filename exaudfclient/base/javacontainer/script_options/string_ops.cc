#include "base/javacontainer/script_options/string_ops.h"
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

namespace StringOps {

void removeQuotesSafely(std::string & s, const std::string & whitespaces) {
    const size_t startQuoteIdx = s.find_first_of("\"'");
    const size_t endQuoteIdx = s.find_last_of("\"'");

    if (startQuoteIdx != std::string::npos && endQuoteIdx != std::string::npos && startQuoteIdx < endQuoteIdx &&
        s[startQuoteIdx] == s[endQuoteIdx]) {
        //Search backwards if there any none whitespace characters in front of quote. If yes, we ignore the quote.
        if (startQuoteIdx > 0) {
            const size_t startingNotWhitespace = s.find_last_not_of(whitespaces, startQuoteIdx-1);
            if (startingNotWhitespace != std::string::npos) {
                return;
            }
        }

        //Search forward if there any none whitespace characters after ending quote. If yes, we ignore the quote.
        if (endQuoteIdx < s.size() -1 ) {
            const size_t trailingNotWhitespace = s.find_first_not_of(whitespaces, endQuoteIdx+1);
            if (trailingNotWhitespace != std::string::npos) {
                return;
            }
        }
        s = s.substr(startQuoteIdx+1, endQuoteIdx-startQuoteIdx-1);
        std::cerr << "DEBUG0 :" << startQuoteIdx << "-" << endQuoteIdx << std::endl;
        std::cerr << "DEBUG1 :" << s << std::endl;
    }
}

} //namespace StringOps


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
