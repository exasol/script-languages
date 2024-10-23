
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/script_options/converter_legacy.h"
#include "base/javacontainer/script_options/converter_v2.h"

using namespace SWIGVMContainers::JavaScriptOptions;


class ConverterJarTest : public ::testing::TestWithParam<std::pair<std::string, std::vector<std::string>>> {};

TEST_P(ConverterJarTest, jar) {
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
            std::make_pair("test.jar:test2.jar", std::vector<std::string>({"test.jar", "test2.jar"})),
            std::make_pair("\"test.jar:test2.jar\"", std::vector<std::string>({"\"test.jar", "test2.jar\""})),
            std::make_pair("t\\:est.jar:test2.jar", std::vector<std::string>({"est.jar", "t\\", "test2.jar"})),
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    ConverterJarTest,
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
            std::make_pair("test.jar:test2.jar", std::vector<std::string>({"test.jar", "test2.jar"})),
            std::make_pair("\"test.jar:test2.jar\"", std::vector<std::string>({"test.jar", "test2.jar"})),
            std::make_pair("\"test .jar:test2.jar\"", std::vector<std::string>({"test .jar", "test2.jar"})),
            std::make_pair("\"test .jar\":test2.jar", std::vector<std::string>({"test .jar", "test2.jar"})),
            std::make_pair("'test .jar':test2.jar", std::vector<std::string>({"test .jar", "test2.jar"})),
            std::make_pair("\"test .jar':test2.jar", std::vector<std::string>({"\"test .jar'", "test2.jar"})),
            std::make_pair("     \"test .jar  '  : ' test2.jar ' \t", std::vector<std::string>({"     \"test .jar  '  ", " test2.jar "})),
            std::make_pair("  abc   \"test .jar  '  : ' test2.jar ' \t", std::vector<std::string>({"  abc   \"test .jar  '  ", " test2.jar "})),
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    ConverterV2JarTest,
    ::testing::ValuesIn(jar_strings_v2)
);
