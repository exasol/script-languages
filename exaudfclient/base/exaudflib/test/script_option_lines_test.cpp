#include "exaudflib/vm/scriptoptionlines.h"
#include <gtest/gtest.h>
#include <string>
#include <exception>

const std::string whitespace = " \t\f\v";
const std::string lineEnd = ";";

class TestException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

void throwException(const char* ex) {
    throw TestException(std::string(ex));
}

using namespace ExecutionGraph;


class ScriptOptionLinesWhitespaceTest : public ::testing::TestWithParam<std::tuple<std::string, std::string, std::string, std::string, std::string>> {};

TEST_P(ScriptOptionLinesWhitespaceTest, WhitespaceExtractOptionLineTest) {
    size_t pos;
    const std::string prefix = std::get<0>(GetParam());
    const std::string suffix = std::get<1>(GetParam());
    const std::string option = std::get<2>(GetParam());
    const std::string value = std::get<3>(GetParam());
    const std::string payload =  std::get<4>(GetParam());
    std::string code = prefix + option + value + lineEnd + suffix + "\n" + payload;
    const std::string res = extractOptionLine(code, option, whitespace, lineEnd, pos, throwException);
    EXPECT_EQ(res, value);
    EXPECT_EQ(code, prefix + suffix + "\n" + payload);
}

std::vector<std::string> white_space_strings = {"", " ", "\t", "\f", "\v", "\n", " \t", "\t ", "\t\f", "\f\t", "\f ", " \f", "\t\v", "\v\t", "\v ", " \v", "\f\v", "\v\f", "  \t", " \t "};
std::vector<std::string> keywords = {"%import", "jvmoption", "%scriptclass", "%jar", "%env"};
std::vector<std::string> values = {"something", "com.mycompany.MyScriptClass", "LD_LIBRARY_PATH=/nvdriver", "-Xms128m -Xmx1024m -Xss512k", "/buckets/bfsdefault/default/my_code.jar"};
std::vector<std::string> payloads = {"anything", "\n\ndef my_func:\n\tpass", "class MyJava\n public static void Main() {\n};\n"};

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesWhitespaceTest,
    ::testing::Combine(::testing::ValuesIn(white_space_strings),
                       ::testing::ValuesIn(white_space_strings),
                       ::testing::ValuesIn(keywords),
                       ::testing::ValuesIn(values),
                       ::testing::ValuesIn(payloads)
    )
);

TEST(ScriptOptionLinesTest, ignore_anything_other_than_whitepsace) {
    size_t pos;
    std::string code =
        "abc %option myoption;\n"
        "\nmycode";
    const std::string res = extractOptionLine(code, "%option", whitespace, lineEnd, pos, throwException);
    EXPECT_TRUE(res.empty());
}

TEST(ScriptOptionLinesTest, need_line_end_character) {
    size_t pos;
    std::string code =
        "%option myoption\n"
        "\nmycode";
   EXPECT_THROW({
        const std::string res = extractOptionLine(code, "%option", whitespace, lineEnd, pos, throwException);
    }, TestException );
}

TEST(ScriptOptionLinesTest, only_finds_the_first_option) {
    size_t pos;
    std::string code =
        "%option myoption; %option mysecondoption;\n"
        "\nmycode";
    const std::string res = extractOptionLine(code, "%option", whitespace, lineEnd, pos, throwException);
    EXPECT_EQ(res, "myoption");
    const std::string expected_resulting_code =
        " %option mysecondoption;\n"
        "\nmycode";

    EXPECT_EQ(code, expected_resulting_code);
}


class ScriptOptionLinesInvalidOptionTest : public ::testing::TestWithParam<std::string> {};


TEST_P(ScriptOptionLinesInvalidOptionTest, value_is_mandatory) {
    size_t pos;
    const std::string invalid_option = GetParam();
    std::string code = invalid_option + "\nsomething";
    EXPECT_THROW({
     const std::string res = extractOptionLine(code, "%option", whitespace, lineEnd, pos, throwException);
    }, TestException );
}

std::vector<std::string> invalid_options = {"%option ;", "%option \n", "\n%option\n;", "%option\nvalue;"};

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesInvalidOptionTest,
    ::testing::ValuesIn(invalid_options)
);

TEST(ScriptOptionLinesTest, ignores_any_other_option) {
    size_t pos;
    const std::string original_code =
        "%option myoption; %option mysecondoption;\n"
        "\nmycode";
    std::string code = original_code;
    const std::string res = extractOptionLine(code, "%mythirdoption", whitespace, lineEnd, pos, throwException);
    EXPECT_TRUE(res.empty());
    EXPECT_EQ(code, original_code);
}

