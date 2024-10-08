#include "base/javacontainer/script_options/checksum.h"
#include <openssl/md5.h>
#include <string.h>

namespace SWIGVMContainers {

namespace JavaScriptOptions {

inline std::vector<unsigned char> scriptToMd5(const char *script) {
    MD5_CTX ctx;
    unsigned char md5[MD5_DIGEST_LENGTH];
    MD5_Init(&ctx);
    MD5_Update(&ctx, script, strlen(script));
    MD5_Final(md5, &ctx);
    return std::vector<unsigned char>(md5, md5 + sizeof(md5));
}


bool Checksum::addScript(const char *script) {
    return m_importedScriptChecksums.insert(scriptToMd5(script)).second;
}


} //namespace JavaScriptOptions

} //namespace SWIGVMContainers
