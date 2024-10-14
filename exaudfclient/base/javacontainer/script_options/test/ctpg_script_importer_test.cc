
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
    /**
       This test checks that running an infinite recursion of the script import will result in the expected exception.
       For that, the test creates new "import scripts" on the fly:
       Whenever the parser finds a new 'import script' option,
       it calls SWIGVMContainers::SWIGMetadataIf::moduleContent().
       The mocked implementation redirects this request to `buildNewScriptCode()` which creates a
       new dummy import script with another '%import ...` option.
     */

    size_t currentIdx = 0;
    std::unique_ptr<SWIGVMContainers::SwigFactory> swigFactory =
            std::make_unique<SwigFactoryTestImpl>([&](const char* scriptKey) {
                        checkIndex(currentIdx, scriptKey);
                        return buildNewScriptCode(++currentIdx);
                       });
    ScriptOptionLinesParserCTPG parser(std::move(swigFactory));

    const std::string code = buildNewScriptCode(currentIdx);
    parser.prepareScriptCode(code);
    EXPECT_THROW({
        try
        {
            parser.extractImportScripts();
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

