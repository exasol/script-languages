
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "javacontainer/script_options/string_ops.h"

using namespace SWIGVMContainers::JavaScriptOptions;




TEST(StringOpsTest, trim) {
    std::string sample = " \tHello World \t";
    StringOps::trim(sample);
    EXPECT_EQ(sample, "Hello World");
}

TEST(StringOpsTest, trimWithNoneASCII) {
    /*
    Test that trim works correctly with None-ASCII characters
    \xa0's bit sequence is '1010 0000', while space bit sequence '0010 0000'.
    If StringOps::trim() would not work correctly with characters where MSB is set, it would interpret \xa0 as space.
    */
    std::string sample = " \t\xa0Hello World\xa0 \t";
    StringOps::trim(sample);
    EXPECT_EQ(sample, "\xa0Hello World\xa0");
}

class ReplaceTrailingEscapeWhitespacesTest : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(ReplaceTrailingEscapeWhitespacesTest, s) {
    const std::pair<std::string, std::string> underTest = GetParam();

    std::string str = underTest.first;
    StringOps::replaceTrailingEscapeWhitespaces(str);
    ASSERT_EQ(str, underTest.second);
}

const std::vector<std::pair<std::string, std::string>> replace_trailing_escape_whitespaces_strings =
        {
            std::make_pair("hello world", std::string("hello world")),
            std::make_pair("hello world ", std::string("hello world")),
            std::make_pair("hello world\\t", std::string("hello world\t")),
            std::make_pair("hello world\\f", std::string("hello world\f")),
            std::make_pair("hello world\\v", std::string("hello world\v")),
            std::make_pair("hello world\\\\t", std::string("hello world\\t")),
            std::make_pair("hello world\\\\t\t", std::string("hello world\\t")),
            std::make_pair("hello world\\\\\\t\t", std::string("hello world\\\t")),
            std::make_pair("hello world\\\\\\\\t\t", std::string("hello world\\\\t")),
            std::make_pair("hello worl\td\\\\\\\\t\t", std::string("hello worl\td\\\\t")),
            std::make_pair("t\t ", std::string("t")),
            std::make_pair("hello worl\td\\\\\\\\", std::string("hello worl\td\\\\\\\\")), //If no whitespace escape sequence, backslashes are not interpreted as backslash escape sequence
        };


INSTANTIATE_TEST_SUITE_P(
    StringOpsTest,
    ReplaceTrailingEscapeWhitespacesTest,
    ::testing::ValuesIn(replace_trailing_escape_whitespaces_strings)
);