#ifndef SCRIPTOPTIONLINEPARSERCHECKSUM_H
#define SCRIPTOPTIONLINEPARSERCHECKSUM_H 1

#include <string>
#include <vector>
#include <set>


namespace SWIGVMContainers {

namespace JavaScriptOptions {

class Checksum {

public:
    Checksum() = default;

    bool addScript(const char *script);

private:
    std::set<std::vector<unsigned char> > m_importedScriptChecksums;
};


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers

#endif //SCRIPTOPTIONLINEPARSERCHECKSUM_H