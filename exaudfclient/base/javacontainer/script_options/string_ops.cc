#include "base/javacontainer/script_options/string_ops.h"
#include <regex>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

namespace StringOps {

inline uint32_t countBackslashesBackwards(const std::string & s, size_t pos) {
    uint32_t retVal(0);
    while (pos >= 0 && s[pos--] == '\\') retVal++;
    return retVal;
}

inline size_t replaceOnlyBackslashSequencesButKeepChar(std::string & s, size_t backslashStartIdx, size_t nBackslashes) {
    const uint32_t nHalfBackslashes = (nBackslashes>>1);
    if (nHalfBackslashes > 0) {
        s = s.erase(backslashStartIdx, nHalfBackslashes );
    }
    const size_t newBackslashEndIdx = backslashStartIdx + nHalfBackslashes;
    return newBackslashEndIdx + 1; //+1 because of the last None-Whitespace character we need to keep
}

inline size_t replaceBackslashSequencesAndWhitespaceSequence(std::string & s, size_t backslashStartIdx,
                                                                size_t nBackslashes, const char* replacement) {
    const uint32_t nHalfBackslashes = (nBackslashes>>1);
    s = s.erase(backslashStartIdx, nHalfBackslashes+1 ); //Delete also backslash of whitespace escape sequence
    const size_t newBackslashEndIdx = backslashStartIdx + nHalfBackslashes;
    s = s.replace(newBackslashEndIdx, 1, replacement);
    return newBackslashEndIdx + 1; //+1 because of the replaced whitespace character
}

inline size_t replaceCharAtPositionAndBackslashes(std::string & s, size_t pos, const char* replacement) {
    const uint32_t nBackslashes = countBackslashesBackwards(s, pos-1);

    const size_t backslashStartIdx = pos-nBackslashes;
    if(nBackslashes % 2 == 0) {
        return replaceOnlyBackslashSequencesButKeepChar(s, backslashStartIdx, nBackslashes);
    }
    else {
        return replaceBackslashSequencesAndWhitespaceSequence(s, backslashStartIdx, nBackslashes, replacement);
    }
}

void replaceTrailingEscapeWhitespaces(std::string & s) {
    if (s.size() > 0) {
        const size_t lastNoneWhitespaceIdx = s.find_last_not_of(" \t\v\f");
        if (lastNoneWhitespaceIdx != std::string::npos) {
            size_t firstWhitespaceAfterNoneWhitespaceIdx = lastNoneWhitespaceIdx + 1;
            if (s.size() > 1) {
                if(s[lastNoneWhitespaceIdx] == 't') {
                    firstWhitespaceAfterNoneWhitespaceIdx = replaceCharAtPositionAndBackslashes(s, lastNoneWhitespaceIdx, "\t");
                } else if (s[lastNoneWhitespaceIdx] == '\\' && s[lastNoneWhitespaceIdx+1] == ' ') {
                    firstWhitespaceAfterNoneWhitespaceIdx = replaceCharAtPositionAndBackslashes(s, lastNoneWhitespaceIdx+1, " ");
                } else if (s[lastNoneWhitespaceIdx] == 'f') {
                    firstWhitespaceAfterNoneWhitespaceIdx = replaceCharAtPositionAndBackslashes(s, lastNoneWhitespaceIdx, "\f");
                } else if (s[lastNoneWhitespaceIdx] == 'v') {
                    firstWhitespaceAfterNoneWhitespaceIdx = replaceCharAtPositionAndBackslashes(s, lastNoneWhitespaceIdx, "\v");
                }
            }
            if (firstWhitespaceAfterNoneWhitespaceIdx != std::string::npos &&
                firstWhitespaceAfterNoneWhitespaceIdx < s.size()) {
                s = s.substr(0, firstWhitespaceAfterNoneWhitespaceIdx); //Right Trim the string
            }
        } else {
            s = "";
        }
    }
}

} //namespace StringOps


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
