
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/script_options/converter_legacy.h"
#include "base/javacontainer/script_options/converter_v2.h"

using namespace SWIGVMContainers::JavaScriptOptions;


class LegacyConverterJarTest : public ::testing::TestWithParam<std::pair<std::string, std::set<std::string>>> {};

TEST_P(LegacyConverterJarTest, jar) {
    const std::pair<std::string, std::set<std::string>> option_value = GetParam();
    const std::string jar_option_value = option_value.first;

    ConverterLegacy converter;
    converter.convertExternalJar(option_value.first);
    ASSERT_EQ(converter.getJarPaths(), option_value.second);
}

const std::vector<std::pair<std::string, std::set<std::string>>> jar_strings =
        {
            std::make_pair("test.jar:test2.jar", std::set<std::string>({"test.jar", "test2.jar"})),
            std::make_pair("\"test.jar:test2.jar\"", std::set<std::string>({"\"test.jar", "test2.jar\""})),
            std::make_pair("t\\:est.jar:test2.jar", std::set<std::string>({"t\\", "est.jar", "test2.jar"})),
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    LegacyConverterJarTest,
    ::testing::ValuesIn(jar_strings)
);



class ConverterV2JarTest : public ::testing::TestWithParam<std::pair<std::string, std::set<std::string>>> {};

TEST_P(ConverterV2JarTest, jar) {
    const std::pair<std::string, std::set<std::string>> option_value = GetParam();
    const std::string jar_option_value = option_value.first;
    std::cerr << "DEBUG: " << jar_option_value << std::endl;

    ConverterV2 converter;
    converter.convertExternalJar(option_value.first);
    ASSERT_EQ(converter.getJarPaths(), option_value.second);
}

const std::vector<std::pair<std::string, std::set<std::string>>> jar_escape_sequences =
        {
            std::make_pair("test.jar:test2.jar", std::set<std::string>({"test.jar", "test2.jar"})),
            std::make_pair("\"test.jar:test2.jar\"", std::set<std::string>({"\"test.jar", "test2.jar\""})),
            std::make_pair("t\\:est.jar:test2.jar", std::set<std::string>({"t\\", "est.jar", "test2.jar"})),
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    ConverterV2JarTest,
    ::testing::ValuesIn(jar_escape_sequences)
);
