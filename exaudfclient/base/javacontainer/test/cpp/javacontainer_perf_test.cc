#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/test/cpp/javavm_test.h"
#include "base/javacontainer/test/cpp/swig_factory_test.h"
#include <string.h>

const uint32_t NumInlineJavaLines = 500000;
const uint32_t NumInlineJavaWordsPerLine = 100;


TEST(JavaContainerPerformance, large_inline_java_udf_test) {
    std::string script_code =
        "%jvmoption option1=abc;\n"
        "%jvmoption option2=def;\n"
        "class JVMOPTION_TEST_WITH_SPACE {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n";

    for (uint32_t idxLine(0); idxLine < NumInlineJavaLines; idxLine++) {
        for (uint32_t idxWord(0); idxWord < NumInlineJavaWordsPerLine; idxWord++)
            script_code.append("somecode ");
        script_code.append("\n");
    }
    script_code.append(" }\n}\n");
    JavaVMTest vm(script_code);
}


