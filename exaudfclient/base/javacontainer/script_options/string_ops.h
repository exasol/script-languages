#ifndef SCRIPTOPTIONLINEPARSERSTRINGOPS_H
#define SCRIPTOPTIONLINEPARSERSTRINGOPS_H 1

#include <string>
#include <algorithm>



namespace SWIGVMContainers {

namespace JavaScriptOptions {

namespace StringOps {

//Following code is based on  https://stackoverflow.com/a/217605

inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

} //namespace StringOps


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERSTRINGOPS_H