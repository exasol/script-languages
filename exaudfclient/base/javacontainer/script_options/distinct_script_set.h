#ifndef SCRIPTOPTIONLINEPARSERCHECKSUM_H
#define SCRIPTOPTIONLINEPARSERCHECKSUM_H 1

#include <string>
#include <vector>
#include <unordered_set>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

class DistinctScriptSet {

public:
    DistinctScriptSet() = default;

    bool addScript(const char *script);

private:
    std::unordered_set<std::string> m_importedScripts;
};


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCHECKSUM_H