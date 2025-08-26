#ifndef SCRIPTOPTIONLINEPARSERCHECKSUM_H
#define SCRIPTOPTIONLINEPARSERCHECKSUM_H 1

#include <string>
#include <vector>
#include <set>
#include <unordered_set>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Checksum {

public:
    Checksum() = default;

    bool addScript(const char *script);

private:
    std::unordered_set<std::string> m_setOfImportedScripts;
};


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCHECKSUM_H