#include "javacontainer/script_options/distinct_script_set.h"
#include <string.h>

namespace SWIGVMContainers {

namespace JavaScriptOptions {


bool DistinctScriptSet::addScript(const char *script) {
    std::string strScript = std::string(script);
    return m_importedScripts.insert(strScript).second;
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
