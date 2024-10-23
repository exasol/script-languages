#ifndef SCRIPTOPTIONLINEPEXTRACTOR_H
#define SCRIPTOPTIONLINEPEXTRACTOR_H 1

#include <string>
#include <vector>
#include <functional>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct Extractor {
    virtual ~Extractor() {}

    virtual void iterateJarPaths(std::function<void(const std::string &option)> callback) const  = 0;

    virtual std::vector<std::string>&& moveJvmOptions() = 0;

    virtual void extract(std::string & scriptCode) = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTOR_H