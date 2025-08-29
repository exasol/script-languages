#include "base/javacontainer/script_options/checksum.h"
#include <string.h>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

/* The following code is dependent on openssl for calculating md5sum
 * md5sum was used to detect if two imported scripts are same
 * As we started to use unordered_set for detecting the same,
 *      the following code and openssl dependency are avoided
inline std::vector<unsigned char> scriptToMd5(const char *script) {
    MD5_CTX ctx;
    unsigned char md5[MD5_DIGEST_LENGTH];
    MD5_Init(&ctx);
    MD5_Update(&ctx, script, strlen(script));
    MD5_Final(md5, &ctx);
    return std::vector<unsigned char>(md5, md5 + sizeof(md5));
}*/


bool Checksum::addScript(const char *script) {
    return m_setOfImportedScripts.insert(std::string(script)).second;
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
