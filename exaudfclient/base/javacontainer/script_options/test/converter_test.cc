
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
    std::cerr << "DEBUG: " << jar_option_value << std::endl;

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
