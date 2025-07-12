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

    for (uint32_t idxLine(0); idxLine < NumInlineJavaLines; ++idxLine) {
        for (uint32_t idxWord(0); idxWord < NumInlineJavaWordsPerLine; ++idxWord)
            script_code.append("somecode ");
        script_code.append("\n");
    }
    script_code.append(" }\n}\n");
    JavaVMTest vm(script_code);
}

TEST(JavaContainerPerformance, large_inline_single_line_full_java_udf_test) {
    std::string script_code =
        "%jvmoption option1=abc;"
        "%jvmoption option2=def;"
        "class JVMOPTION_TEST_WITH_SPACE {"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {";

    for (uint32_t idxLine(0); idxLine < NumInlineJavaLines; ++idxLine) {
        for (uint32_t idxWord(0); idxWord < NumInlineJavaWordsPerLine; ++idxWord)
            script_code.append("somecode ");

    }
    script_code.append(" }}");
    JavaVMTest vm(script_code);
}

TEST(JavaContainerPerformance, large_inline_single_line_slim_java_udf_test) {
    std::string script_code =
        "%jvmoption option1=abc;"
        "%jvmoption option2=def;"
        "class JVMOPTION_TEST_WITH_SPACE {"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {";

    for (uint32_t idxLine(0); idxLine < NumInlineJavaLines / 10; ++idxLine) {
        for (uint32_t idxWord(0); idxWord < NumInlineJavaWordsPerLine / 10; ++idxWord)
            script_code.append("someco%de ; \\t");

    }
    script_code.append(" }}");
    JavaVMTest vm(script_code);
}

