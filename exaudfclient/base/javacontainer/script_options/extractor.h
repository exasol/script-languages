#ifndef SCRIPTOPTIONLINEPEXTRACTOR_H
#define SCRIPTOPTIONLINEPEXTRACTOR_H 1

#include <string>
#include <vector>
#include <set>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

struct Extractor {
    virtual ~Extractor() {}

    virtual const std::set<std::string> & getJarPaths() const  = 0;

    virtual std::vector<std::string>&& moveJvmOptions() = 0;

    virtual void extract(std::string & scriptCode) = 0;
};

} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPEXTRACTOR_H