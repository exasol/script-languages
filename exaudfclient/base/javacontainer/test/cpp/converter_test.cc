
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/script_options/converter.h"

using namespace SWIGVMContainers::JavaScriptOptions;


class ConverterJarTest : public ::testing::TestWithParam<std::pair<std::string, std::set<std::string>>> {};

TEST_P(ConverterJarTest, jar) {
    const std::pair<std::string, std::set<std::string>> option_value = GetParam();
    const std::string jar_option_value = option_value.first;

    Converter converter;
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
    ConverterJarTest,
    ::testing::ValuesIn(jar_strings)
);



class ConverterJarEscapeSequenceTest : public ::testing::TestWithParam<std::pair<std::string, std::set<std::string>>> {};

TEST_P(ConverterJarEscapeSequenceTest, jar) {
    const std::pair<std::string, std::set<std::string>> option_value = GetParam();
    const std::string jar_option_value = option_value.first;
    std::cerr << "DEBUG: " << jar_option_value << std::endl;

    Converter converter;
    converter.convertExternalJarWithEscapeSequences(option_value.first);
    ASSERT_EQ(converter.getJarPaths(), option_value.second);
}

const std::vector<std::pair<std::string, std::set<std::string>>> jar_escape_sequences =
        {
            std::make_pair("test.jar:test2.jar", std::set<std::string>({"test.jar", "test2.jar"})),
            std::make_pair("\"test.jar:test2.jar\"", std::set<std::string>({"test.jar", "test2.jar"})),
            std::make_pair("\"test.jar:test2.jar", std::set<std::string>({"\"test.jar", "test2.jar"})),
            std::make_pair("t\\:est.jar:test2.jar", std::set<std::string>({"t\\:est.jar", "test2.jar"})),
            std::make_pair("\\:test.jar:test2.jar", std::set<std::string>({"\\:test.jar", "test2.jar"})),
            std::make_pair("test.jar\\\\:test2.jar", std::set<std::string>({"test.jar\\\\", "test2.jar"})),
            std::make_pair("test.jar\\\\\\:", std::set<std::string>({"test.jar\\\\\\:"})),
        };

INSTANTIATE_TEST_SUITE_P(
    Converter,
    ConverterJarEscapeSequenceTest,
    ::testing::ValuesIn(jar_escape_sequences)
);
