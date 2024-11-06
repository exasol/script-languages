
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/script_options/converter_legacy.h"
#include "base/javacontainer/script_options/converter_v2.h"

using namespace SWIGVMContainers::JavaScriptOptions;


class LegacyConverterJarTest : public ::testing::TestWithParam<std::pair<std::string, std::vector<std::string>>> {};

TEST_P(LegacyConverterJarTest, jar) {
    const std::pair<std::string, std::vector<std::string>> option_value = GetParam();
    const std::string jar_option_value = option_value.first;

    ConverterLegacy converter;
    converter.convertExternalJar(option_value.first);
    std::vector<std::string> result;
    converter.iterateJarPaths([&](auto jar) {result.push_back(jar);});
    ASSERT_EQ(result, option_value.second);
}

const std::vector<std::pair<std::string, std::vector<std::string>>> jar_strings =
        {
            std::make_pair("test.jar:test2.jar", std::vector<std::string>({"test.jar", "test2.jar"})), //basic splitting
            std::make_pair("test.jar:test.jar", std::vector<std::string>({"test.jar"})), //filter duplicates
            std::make_pair("testDEF.jar:testABC.jar", std::vector<std::string>({"testABC.jar", "testDEF.jar"})), //alphabetical order
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    LegacyConverterJarTest,
    ::testing::ValuesIn(jar_strings)
);



class ConverterV2JarTest : public ::testing::TestWithParam<std::pair<std::string, std::vector<std::string>>> {};

TEST_P(ConverterV2JarTest, jar) {
    const std::pair<std::string, std::vector<std::string>> option_value = GetParam();
    const std::string jar_option_value = option_value.first;

    ConverterV2 converter;
    converter.convertExternalJar(option_value.first);
    std::vector<std::string> result;
    converter.iterateJarPaths([&](auto jar) {result.push_back(jar);});
    ASSERT_EQ(result, option_value.second);
}

const std::vector<std::pair<std::string, std::vector<std::string>>> jar_strings_v2 =
        {
            std::make_pair("test.jar:test2.jar", std::vector<std::string>({"test.jar", "test2.jar"})), //basic splitting
            std::make_pair("test.jar:test.jar", std::vector<std::string>({"test.jar", "test.jar"})), //keep duplicates
            std::make_pair("testDEF.jar:testABC.jar", std::vector<std::string>({"testDEF.jar", "testABC.jar"})), //maintain order
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    ConverterV2JarTest,
    ::testing::ValuesIn(jar_strings_v2)
);

class ConverterV2JvmOptionsTest : public ::testing::TestWithParam<std::pair<std::string, std::vector<std::string>>> {};

TEST_P(ConverterV2JvmOptionsTest, jvm_option) {
    const std::pair<std::string, std::vector<std::string>> option_value = GetParam();
    const std::string jvm_option_value = option_value.first;

    ConverterV2 converter;
    converter.convertJvmOption(option_value.first);
    std::vector<std::string> result = std::move(converter.moveJvmOptions());
    ASSERT_EQ(result, option_value.second);
}

const std::vector<std::pair<std::string, std::vector<std::string>>> jvm_options_strings_v2 =
        {
            std::make_pair("optionA=abc optionB=def", std::vector<std::string>({"optionA=abc", "optionB=def"})),
            std::make_pair("optionA=abc\\ def optionB=ghi", std::vector<std::string>({"optionA=abc def", "optionB=ghi"})),
            std::make_pair("optionA=abc\\tdef optionB=ghi", std::vector<std::string>({"optionA=abc\tdef", "optionB=ghi"})),
            std::make_pair("   optionA=abc\\tdef optionB=ghi", std::vector<std::string>({"optionA=abc\tdef", "optionB=ghi"})),
            std::make_pair("   optionA=abc\\tdef\\\\\t\t optionB=ghi", std::vector<std::string>({"optionA=abc\tdef\\", "optionB=ghi"})),
            std::make_pair("   optionA=abc\\tdef\\\\\\t\\t optionB=ghi", std::vector<std::string>({"optionA=abc\tdef\\\t\t", "optionB=ghi"})),
            std::make_pair("   optionA=abc\\tdef\\\\\\t\\t optionB=ghi   ", std::vector<std::string>({"optionA=abc\tdef\\\t\t", "optionB=ghi"}))
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    ConverterV2JvmOptionsTest,
    ::testing::ValuesIn(jvm_options_strings_v2)
);
