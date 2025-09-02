#include "base/javacontainer/script_options/checksum.h"
#include <string.h>

namespace SWIGVMContainers {

namespace JavaScriptOptions {


bool Checksum::addScript(const char *script) {
    std::string strScript = std::string(script);
    return m_importedScriptChecksums.insert(strScript).second;
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
