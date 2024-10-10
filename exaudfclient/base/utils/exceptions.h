#ifndef UTILS_EXCEPTION_H
#define UTILS_EXCEPTION_H

#include <string_view>
#include <string>

namespace Utils {

template<typename TException>
inline void rethrow(const TException& sourceException, const std::string_view& prefix) {
    throw TException(std::string(prefix) + " " + sourceException.what());
}

} //namespace Utils

#endif //UTILS_EXCEPTION_H