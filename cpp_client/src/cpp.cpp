#include "cpp.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <unistd.h>
#include "swigcontainers_ext.h"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <zmq.hpp>
#include <fcntl.h>
#include <fstream>
#include <linux/limits.h>
#include <openssl/md5.h>
#include <dlfcn.h>
#include <scriptDTO.h>

namespace SWIGVMContainers {
std::string getExecutablePath()
{
    char buf[PATH_MAX+1];
    ssize_t count = readlink("/proc/self/exe", buf, PATH_MAX);
    if (count>0)
    {
        buf[count] = '\0';
        return std::string(buf);
    }
    abort();
}

bool mexec(const std::string& cmd_, std::string& result) {
    char buffer[128];
    std::stringstream cmd;
    cmd << "ulimit -v 500000; ";
    cmd << cmd_ << " 2>&1";

    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        result = "Cannot start command `" + cmd.str() + "`";
        return false;
    }
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL) {
            result += buffer;
        }
    }
    int s = pclose(pipe);
    if (s == -1)
    {
        return false;
    }
    if (WEXITSTATUS(s))
    {
        return false;
    }
    return true;
}

std::string CPPVM::getOptionLine(std::string scriptCode, const std::string option, const std::string whitespace, const std::string lineEnd, size_t& pos) {
    std::string result;
    size_t startPos = scriptCode.find(option);
    if (startPos != std::string::npos) {
        size_t firstPos = startPos + option.length();
        firstPos = scriptCode.find_first_not_of(whitespace, firstPos);
        if (firstPos == std::string::npos) {
            std::stringstream ss;
            ss << "No values found for " << option << " statement";
            throw exception(ss.str().c_str());
        }
        size_t lastPos = scriptCode.find_first_of(lineEnd + "\r\n", firstPos);
        if (lastPos == std::string::npos || scriptCode.compare(lastPos, lineEnd.length(), lineEnd) != 0) {
            std::stringstream ss;
            ss << "End of " << option << " statement not found";
            throw exception(ss.str().c_str());
        }
        if (firstPos >= lastPos) {
            std::stringstream ss;
            ss << "No values found for " << option << " statement";
            throw exception(ss.str().c_str());
        }
        size_t optionsEnd = scriptCode.find_last_not_of(whitespace, lastPos - 1);
        if (optionsEnd == std::string::npos || optionsEnd < firstPos) {
            std::stringstream ss;
            ss << "No values found for " << option << " statement";
            throw exception(ss.str().c_str());
        }
        result = scriptCode.substr(firstPos, optionsEnd - firstPos + 1);
        scriptCode.erase(startPos, lastPos - startPos + 1);
    }
    pos = startPos;
    return result;
}

std::vector<unsigned char> CPPVM::scriptToMd5(const char *script) {
    MD5_CTX ctx;
    unsigned char md5[MD5_DIGEST_LENGTH];
    MD5_Init(&ctx);
    MD5_Update(&ctx, script, strlen(script));
    MD5_Final(md5, &ctx);
    return std::vector<unsigned char>(md5, md5 + sizeof(md5));
}

void CPPVM::importScripts() {

    const std::string whitespace = " \t\f\v";
    const std::string lineEnd = ";";
    size_t pos;

    // Attention: We must hash the parent script before modifying it (adding the
    // package definition). Otherwise we don't recognize if the script imports itself
    m_importedScriptChecksums.insert(scriptToMd5(meta.scriptCode()));
    while (true) {
        std::string scriptName = getOptionLine(meta.scriptCode(), "%import", whitespace, lineEnd, pos);
        if (scriptName == "")
            break;

        const char *scriptCode = meta.moduleContent(scriptName.c_str());
        const char *exception = meta.checkException();
        if (exception)
            throw SWIGVM::exception(exception);
        if (m_importedScriptChecksums.insert(scriptToMd5(scriptCode)).second) {
            // Script has not been imported yet
            // If this imported script contains %import statements
            // they will be resolved in this while loop.
            m_script_code.insert(pos, scriptCode);
        }
    }
}



CPPVM::CPPVM(bool)
    :
      m_importedScriptChecksums(),
      m_script_code(SWIGVM_params->script_code)
{
    std::string myPath = getExecutablePath();
    std::string myFolder = myPath.substr(0,myPath.find_last_of('/'));
    {
        std::stringstream cmd;
        cmd << "cp  " << myFolder << "/*.h /tmp/";

        if (::system(cmd.str().c_str()))
        {
            std::cerr << "Some error when copying header file" << std::endl;
            std::cerr << "current dir: " << std::endl;
            if (system("pwd")) {}
            abort();
        }
    }
    importScripts();
    const std::string whitespace = " \t\f\v";
    const std::string lineEnd = ";";
    size_t nextOptionPos = 0;

    std::string LDFLAGS = getOptionLine(meta.scriptCode(),"%compilerflags",whitespace,lineEnd,nextOptionPos);

    std::ofstream out("/tmp/code.cc");
    out << "#include \"swigcontainers_ext.h\"" << std::endl;
    out << "#include \"scriptDTOWrapper.h\"" << std::endl;
    out << "using namespace SWIGVMContainers;\n" << std::endl;

    out << m_script_code << std::endl;
    out.close();

    {
        std::stringstream cmd;
        cmd << "g++ -O3 -shared -fPIC -o /tmp/libcode.so /tmp/code.cc -DBUILDINSWIGDIR";
        cmd << " -I" << myFolder;
        cmd << " " << LDFLAGS;

        std::string msg;
        if (!mexec(cmd.str(), msg))
        {

            throw exception(("Error when compiling script code:\n"+cmd.str()+"\n\n"+msg).c_str());
        }
    }

// enable to retrieve function signatures from the EXASOL log file
#if 0
    {
       if (::system("nm /tmp/libcode.so")) {}
    }
#endif
    handle = dlopen("/tmp/libcode.so",RTLD_NOW);

    if (handle == NULL)
    {
        throw exception( dlerror() );
    };


}

void CPPVM::shutdown()
{
}

bool CPPVM::run()
{
    if (SWIGVM_params->singleCallMode)
    {
        throw exception("calling RUN in single call mode");
    }


    if (!run_cpp) {
        char *error;        
        run_cpp = (RUN_FUNC)dlsym(handle, "_Z7run_cppRN16SWIGVMContainers12SWIGMetadataERNS_17SWIGTableIteratorERNS_17SWIGResultHandlerE");
        if ((error = dlerror()) != NULL)  {
            std::stringstream sb;
            sb << "Error when trying to load function \"run_cpp\": " << std::endl << error;
            throw exception(sb.str().c_str());
        }
    }

    SWIGTableIterator iter;
    SWIGResultHandler res(&iter);

    if (meta.inputType()==MULTIPLE) {
        // please note: at the moment, SET-RETURNS and SET-EMITS call the same
        // C++ function.
        if (meta.outputType()==EXACTLY_ONCE) {
            // SET-RETURNS
            (*run_cpp)(meta,iter,res);
        } else {
            // SET-EMITS
            (*run_cpp)(meta,iter,res);
        }
    } else {
        // please note: at the moment, SCALAR-RETURNS and SCALAR-EMITS call the same
        // C++ function.
        if (meta.outputType()==EXACTLY_ONCE) {
            // SCALAR-RETURNS
            while (true) {
                (*run_cpp)(meta,iter,res);
                if (!iter.next()) break;
            }
        } else {
            // SCALAR-EMITS
            while (true) {
                (*run_cpp)(meta,iter,res);
                if (!iter.next()) break;
            }
        }
    }

    res.flush();


    return true;
}

std::string CPPVM::singleCall(single_call_function_id fn, const ExecutionGraph::ScriptDTO& args)
{
    DEFAULT_OUTPUT_COLUMNS_FUNC defaultOutputColumnsFunc = NULL;
    ADAPTER_CALL_FUNC adapterCallFunc = NULL;
    IMPORT_ALIAS_FUNC importAliasFunc = NULL;
    ExecutionGraph::StringDTO* stringDTO = NULL;
    ExecutionGraph::ImportSpecification* importDTO = NULL;
    char *error = NULL;
    switch (fn)
    {
    case SC_FN_DEFAULT_OUTPUT_COLUMNS:
        defaultOutputColumnsFunc = (DEFAULT_OUTPUT_COLUMNS_FUNC)dlsym(handle, "_Z23getDefaultOutputColumnsB5cxx11RKN9UDFClient8MetadataE");
        if ((error = dlerror()) != NULL)
        {
            std::stringstream sb;
            sb << "Error when trying to load singleCall function: " << std::endl << error;
            throw exception(sb.str().c_str());
        }
        return (*defaultOutputColumnsFunc)(meta);
        break;
    case SC_FN_VIRTUAL_SCHEMA_ADAPTER_CALL:
        adapterCallFunc = (ADAPTER_CALL_FUNC)dlsym(handle, "_Z11adapterCallRKN9UDFClient8MetadataENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
        if ((error = dlerror()) != NULL)
        {
            std::stringstream sb;
            sb << "Error when trying to load singleCall function: " << std::endl << error;
            throw exception(sb.str().c_str());
        }
        stringDTO = (ExecutionGraph::StringDTO*)&args;
        assert(stringDTO != NULL);
        return (*adapterCallFunc)(meta,stringDTO->getArg());
        break;
    case SC_FN_GENERATE_SQL_FOR_IMPORT_SPEC:
        importAliasFunc = (IMPORT_ALIAS_FUNC)dlsym(handle,"_Z24generateSqlForImportSpecB5cxx11RKN9UDFClient8MetadataERKNS_19ImportSpecificationE");
        if ((error = dlerror()) != NULL)
        {
            std::stringstream sb;
            sb << "Error when trying to load singleCall function: " << std::endl << error;
            throw exception(sb.str().c_str());
        }
        importDTO = (ExecutionGraph::ImportSpecification*)&args;
        assert(importDTO != NULL);
        return (*importAliasFunc)(meta,*importDTO);
        break;
    default:
    {
        std::stringstream sb;
        sb << "Unsupported singleCall function id: " << fn;
        throw exception(sb.str().c_str());
    }
    }

    return "dummy";
}



}
