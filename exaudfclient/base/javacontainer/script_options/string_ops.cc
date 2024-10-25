#include "base/javacontainer/script_options/string_ops.h"
#include <regex>
#include <iostream>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

namespace StringOps {

inline uint32_t countBackslashesBackwards(const std::string & s, size_t pos) {
    uint32_t retVal(0);
    while (pos >= 0 && s[pos--] == '\\') retVal++;
    return retVal;
}

inline size_t replaceCharAtPositionAndBackslashes(std::string & s, size_t pos, const char* replacement) {
    std::cerr << "Called replaceCharAtPositionAndBackslashes with pos=" << pos << " replacement='" << replacement << "'" << std::endl;
    const uint32_t nBackslashes = countBackslashesBackwards(s, pos-1);
    std::cerr << "nBackslashes=" << nBackslashes << std::endl;
    size_t rtrimIdx = std::string::npos;
    if(nBackslashes % 2 == 0) {
        //<replacement> does not belong to an escape sequence
        //Delete half of the backslashes because they belong to the escape sequences
        if (nBackslashes > 0) {
            s = s.erase(pos-nBackslashes, (nBackslashes>>1) );
        }
        rtrimIdx = pos + 1 - (nBackslashes>>1);
    }
    else {
        //<replacement> does belong to an escape sequence
        //Delete half of the backslashes because they belong to the escape sequences + 1 of the <replacement>
        s = s.erase(pos-nBackslashes, (nBackslashes>>1)+1 );
        s = s.replace(pos - (nBackslashes>>1) - 1, 1, replacement);
        rtrimIdx = pos - (nBackslashes>>1);
    }
    return rtrimIdx;
}

void replaceTrailingEscapeWhitespaces(std::string & s) {
    if (s.size() > 0) {
        const size_t lastIdx = s.find_last_not_of(" \t\v\f");
        if (lastIdx != std::string::npos) {
            size_t rtrimIdx = lastIdx + 1;
            if (s.size() > 1) {
                if(s[lastIdx] == 't') {
                    rtrimIdx = replaceCharAtPositionAndBackslashes(s, lastIdx, "\t");
                } else if (s[lastIdx] == '\\' && s[lastIdx+1] == ' ') {
                    rtrimIdx = replaceCharAtPositionAndBackslashes(s, lastIdx+1, " ");
                } else if (s[lastIdx] == 'f') {
                    rtrimIdx = replaceCharAtPositionAndBackslashes(s, lastIdx, "\f");
                } else if (s[lastIdx] == 'v') {
                    rtrimIdx = replaceCharAtPositionAndBackslashes(s, lastIdx, "\v");
                }
            }
            if (rtrimIdx != std::string::npos && rtrimIdx < s.size()) {
                s = s.substr(0, rtrimIdx);
            }
        } else {
            s = "";
        }
    }
}

} //namespace StringOps


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
