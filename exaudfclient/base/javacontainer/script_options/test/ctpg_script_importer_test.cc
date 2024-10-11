
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/script_options/parser_ctpg.h"

#include "base/javacontainer/script_options/test/swig_factory_test.h"
#include <string.h>

using namespace SWIGVMContainers::JavaScriptOptions;


static const char* sc_scriptName = "script";


void checkIndex(size_t currentIdx, const char* scriptKey) {
    std::stringstream ss;
    ss << sc_scriptName << currentIdx;
    if (ss.str() != scriptKey) {
        throw std::logic_error(std::string("Script Key does not match: '") + ss.str() + " != '" + scriptKey + "'");
    }
}

const char* buildNewScriptCode(size_t currentIdx) {
    std::stringstream ss;
    ss << "%import " << sc_scriptName << currentIdx << ";something";
    static std::string ret;
    ret = ss.str();
    return ret.c_str();
}

TEST(ScriptImporterTest, max_recursion_depth) {

    size_t currentIdx = 0;
    SwigFactoryTestImpl swigFactoryTest([&](const char* scriptKey) {
                        checkIndex(currentIdx, scriptKey);
                        return buildNewScriptCode(++currentIdx);
                       });
    ScriptOptionLinesParserCTPG parser;

    const std::string code = buildNewScriptCode(currentIdx);
    parser.prepareScriptCode(code);
    EXPECT_THROW({
        try
        {
            parser.extractImportScripts(swigFactoryTest);
        }
        catch( const std::runtime_error& e )
        {
            //We need to deceive "find_duplicate_error_codes.sh" here
            const std::string expectedError =
                std::string("F-UDF-CL-SL-JAVA-") + "1633: Maximal recursion depth for importing scripts reached.";
            EXPECT_STREQ( expectedError.c_str(), e.what());
            throw;
        }
    }, std::runtime_error );
}

