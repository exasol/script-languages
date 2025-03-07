#include "base/script_options_parser/ctpg/script_option_lines_ctpg.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <string>
#include <exception>
#include "base/script_options_parser/exception.h"


using namespace ExecutionGraph;
using namespace ExecutionGraph::OptionsLineParser::CTPG;

using ::testing::MatchesRegex;


inline ScriptOption buildOption(const char* value, size_t idx, size_t len) {
    ScriptOption option = { .value = value, .idx_in_source = idx, .size = len};
    return option;
}

#ifndef VALGRIND_ACTIVE

class ScriptOptionLinesWhitespaceTest : public ::testing::TestWithParam<std::tuple<std::string, std::string, std::string, std::string, std::string, std::string, std::string>> {};

TEST_P(ScriptOptionLinesWhitespaceTest, WhitespaceExtractOptionLineTest) {
    const std::string prefix = std::get<0>(GetParam());
    const std::string suffix = std::get<1>(GetParam());
    const std::string new_line = std::get<2>(GetParam());
    const std::string option = std::get<3>(GetParam());
    const std::string delimeter = std::get<4>(GetParam());
    const std::string value = std::get<5>(GetParam());
    const std::string payload =  std::get<6>(GetParam());
    const std::string code = prefix + '%' + option + delimeter + value + ';' + suffix + new_line + payload;
    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 1);
    const auto option_result = result.find(option);
    ASSERT_NE(option_result, result.end());
    ASSERT_EQ(option_result->second.size(), 1);
    EXPECT_EQ(option_result->second[0].value, value);
}

std::vector<std::string> prefixes = {"", " ", "\t", "\f", "\v", "\n", "\r\n", " \t", "\t ", "\t\f", "\f\t", "\f ", " \f", "\t\v", "\v\t", "\v ", " \v", "\f\v", "\v\f", "  \t", " \t "}; //"" for case if there is prefix
std::vector<std::string> suffixes = {"", " ", "\t", "\f", "\v"}; //"" for case if there is suffix
std::vector<std::string> new_lines = {"", "\n", "\r", "\r\n"}; //"" for case if there is no newline
std::vector<std::string> delimeters = {" ", "\t", "\f", "\v", " \t", "\t ", "\t\f", "\f\t", "\f ", " \f", "\t\v", "\v\t", "\v ", " \v", "\f\v", "\v\f", "  \t", " \t "};
std::vector<std::string> keywords = {"import", "jvmoption", "scriptclass", "jar", "env"};
std::vector<std::string> values = {"something", "com.mycompany.MyScriptClass", "LD_LIBRARY_PATH=/nvdriver", "-Xms128m -Xmx1024m -Xss512k", "/buckets/bfsdefault/default/my_code.jar", "something "};
std::vector<std::string> payloads = {"anything", "\n\ndef my_func:\n\tpass", "class MyJava\n public static void Main() {\n};\n"};

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesWhitespaceTest,
    ::testing::Combine(::testing::ValuesIn(prefixes),
                       ::testing::ValuesIn(suffixes),
                       ::testing::ValuesIn(new_lines),
                       ::testing::ValuesIn(keywords),
                       ::testing::ValuesIn(delimeters),
                       ::testing::ValuesIn(values),
                       ::testing::ValuesIn(payloads)
    )
);

#endif //VALGRIND_ACTIVE

TEST(ScriptOptionLinesTest, ignore_anything_other_than_whitepsace) {
    const std::string code =
        "abc %option myoption;\n"
        "\nmycode";
    options_map_t result;
    parseOptions(code, result);
    EXPECT_TRUE(result.empty());
}

TEST(ScriptOptionLinesTest, need_option_termination_character) {
    const std::string code =
        "%option myoption\n"
        "\nmycode";
    options_map_t result;
    EXPECT_THROW({
        try
        {
            parseOptions(code, result);
        }
        catch( const OptionParserException& e )
        {
            // and this tests that it has the correct message
            EXPECT_STREQ( e.what(), "Error parsing script options at line 0: [1:17] PARSE: Syntax error: Unexpected '<eof>'\n");
            throw;
        }
    }, OptionParserException );
}

TEST(ScriptOptionLinesTest, need_option_termination_character_second_line) {
    const std::string code =
        "%optionA myoption;\n"
        "%optionB myoption\n"
        "\nmycode";
    options_map_t result;
    EXPECT_THROW({
        try
        {
            parseOptions(code, result);
        }
        catch( const OptionParserException& e )
        {
            // and this tests that it has the correct message
            EXPECT_STREQ( e.what(), "Error parsing script options at line 1: [1:18] PARSE: Syntax error: Unexpected '<eof>'\n");
            throw;
        }
    }, OptionParserException );
}

TEST(ScriptOptionLinesTest, finds_the_two_options_same_key) {
    const std::string code =
        "%some_option myoption; %some_option mysecondoption;\n"
        "\nmycode";
    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 1);
    const auto option_result = result.find("some_option");

    ASSERT_NE(option_result, result.end());
    ASSERT_EQ(option_result->second.size(), 2);
    ASSERT_EQ(option_result->second[0], buildOption("myoption", 0, 22));
    ASSERT_EQ(option_result->second[1], buildOption("mysecondoption", 23, 28));
}

TEST(ScriptOptionLinesTest, finds_the_two_options_different_keys) {
    const std::string code =
        "%some_option myoption; %otheroption mysecondoption;\n"
        "\nmycode";
    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 2);
    const auto option_result = result.find("some_option");

    ASSERT_NE(option_result, result.end());
    ASSERT_EQ(option_result->second.size(), 1);
    ASSERT_EQ(option_result->second[0], buildOption("myoption", 0, 22));

    const auto otheroption_result = result.find("otheroption");

    ASSERT_NE(otheroption_result, result.end());
    ASSERT_EQ(otheroption_result->second.size(), 1);
    ASSERT_EQ(otheroption_result->second[0], buildOption("mysecondoption", 23, 28));
}

class ScriptOptionLinesInvalidOptionTest : public ::testing::TestWithParam<std::string> {};


TEST_P(ScriptOptionLinesInvalidOptionTest, value_is_mandatory) {
    const std::string invalid_option = GetParam();
    const std::string code = invalid_option + "\nsomething";
    options_map_t result;
    EXPECT_THROW({
        try
        {
            parseOptions(code, result);
        }
        catch( const OptionParserException& e )
        {
            EXPECT_THAT( e.what(), MatchesRegex("^Error parsing script options.*PARSE: Syntax error: Unexpected.*$"));
            throw;
        }
    }, OptionParserException );
}

const std::vector<std::string> invalid_options = {"%some_option ;", "%some_option \n", "\n%some_option\n;", "%some_option\nvalue;"};

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesInvalidOptionTest,
    ::testing::ValuesIn(invalid_options)
);


TEST(ScriptOptionLinesTest, test_when_two_options_plus_code_in_same_line_then_options_parsed_successfully) {
    /**
    Verify the correct behavior of new parser for situation as described in https://github.com/exasol/script-languages-release/issues/652.
    */
    const std::string code = "%jar /buckets/bucketfs1/jars/exajdbc.jar; %jvmoption -Xms4m; class JAVA_UDF_3 {static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {String host_name = ctx.getString(\"col1\");}}\n/\n;";
    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 2);

    const auto jar_option_result = result.find("jar");
    ASSERT_NE(jar_option_result, result.end());
    ASSERT_EQ(jar_option_result->second.size(), 1);
    ASSERT_EQ(jar_option_result->second[0], buildOption("/buckets/bucketfs1/jars/exajdbc.jar", 0, 41));

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_NE(jvm_option_result, result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    ASSERT_EQ(jvm_option_result->second[0], buildOption("-Xms4m", 42, 18));
}


TEST(ScriptOptionLinesTest, test_values_can_contain_spaces) {
    /**
    Verify assumptions as described in https://github.com/exasol/script-languages-release/issues/878
    The parser is actually correct, but the client code incorrectly parses the result (see javacontainer_test.cc - quoted_jvm_option)
    */
    const std::string code =
        "%jvmoption -Dhttp.agent=\"ABC DEF\";\n\n"
        "class JVMOPTION_TEST_WITH_SPACE {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 1);

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_NE(jvm_option_result, result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    ASSERT_EQ(jvm_option_result->second[0], buildOption("-Dhttp.agent=\"ABC DEF\"", 0, 34));
}

TEST(ScriptOptionLinesTest, test_multiple_lines_with_code) {
    /**
    Verify that the parser can read options coming after some code.
    */
    const std::string code =
        "%jvmoption -Dhttp.agent=\"ABC DEF\"; class Abc{};\n\n"
        "%jar /buckets/bucketfs1/jars/exajdbc.jar; class DEF{};\n";

    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 2);

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_NE(jvm_option_result, result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    ASSERT_EQ(jvm_option_result->second[0], buildOption("-Dhttp.agent=\"ABC DEF\"", 0, 34));

    const auto jar_option_result = result.find("jar");
    ASSERT_NE(jar_option_result, result.end());
    ASSERT_EQ(jar_option_result->second.size(), 1);
    ASSERT_EQ(jar_option_result->second[0], buildOption("/buckets/bucketfs1/jars/exajdbc.jar", 49, 41));
}


class ScriptOptionLinesEscapeSequenceTest : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(ScriptOptionLinesEscapeSequenceTest, test_escape_seq_in_option_value) {
    const std::pair<std::string, std::string> option_value = GetParam();
    /**
    Verify that the parser replaces escape sequences correctly.
    */
    const std::string code =
        "%jvmoption " + option_value.first + "; class Abc{};\n"
        "%jar /buckets/bucketfs1/jars/exajdbc.jar; class DEF{};\n";

    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 2);

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_NE(jvm_option_result, result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    EXPECT_EQ(jvm_option_result->second[0].value, option_value.second);

    const auto jar_option_result = result.find("jar");
    ASSERT_NE(jar_option_result, result.end());
    ASSERT_EQ(jar_option_result->second.size(), 1);
    ASSERT_EQ(jar_option_result->second[0].value, "/buckets/bucketfs1/jars/exajdbc.jar");
}

/*
 '\n' -> new line character
 '\r' -> return character
 '\;' -> semicolon
 '\\' -> backslash
 '\ ' or '\t' or '\f' or '\v' at start of option value -> replaced by the respective white space character
 '\ ' or '\t' or '\f' or '\v' in the middle of option value -> should not be replaced
 '\a' -> anything else should not be replaced.
 */
const std::vector<std::pair<std::string, std::string>> escape_sequences =
        {
            std::make_pair("-Dhttp.agent=ABC\\nDEF", "-Dhttp.agent=ABC\nDEF"),
            std::make_pair("-Dhttp.agent=ABC\\rDEF", "-Dhttp.agent=ABC\rDEF"),
            std::make_pair("-Dhttp.agent=ABC\\;DEF", "-Dhttp.agent=ABC;DEF"),
            std::make_pair("-Dhttp.agent=ABC\\\\rDEF", "-Dhttp.agent=ABC\\rDEF"),
            std::make_pair("-Dhttp.agent=ABC\\aDEF", "-Dhttp.agent=ABC\\aDEF"), //any other escape sequence must stay as is
            std::make_pair("\\n-Dhttp.agent=ABCDEF", "\n-Dhttp.agent=ABCDEF"),
            std::make_pair("\\r-Dhttp.agent=ABCDEF", "\r-Dhttp.agent=ABCDEF"),
            std::make_pair("\\;-Dhttp.agent=ABCDEF", ";-Dhttp.agent=ABCDEF"),
            std::make_pair("\\\\r-Dhttp.agent=ABCDEF", "\\r-Dhttp.agent=ABCDEF"),
            std::make_pair("-Dhttp.agent=ABCDEF\\n", "-Dhttp.agent=ABCDEF\n"),
            std::make_pair("-Dhttp.agent=ABCDEF\\r", "-Dhttp.agent=ABCDEF\r"),
            std::make_pair("-Dhttp.agent=ABCDEF\\;", "-Dhttp.agent=ABCDEF;"),
            std::make_pair("-Dhttp.agent=ABCDEF\\\\;", "-Dhttp.agent=ABCDEF\\"),
            std::make_pair("-Dhttp.agent=ABCDEF\\\\\\;", "-Dhttp.agent=ABCDEF\\;"),
            std::make_pair("-Dhttp.agent=ABC\\ DEF", "-Dhttp.agent=ABC\\ DEF"), //escaped white space in middle of string must stay as is
            std::make_pair("\\ -Dhttp.agent=ABCDEF", " -Dhttp.agent=ABCDEF"),
            std::make_pair("\\  \t -Dhttp.agent=ABCDEF", "  \t -Dhttp.agent=ABCDEF"),
            std::make_pair("\\t-Dhttp.agent=ABCDEF", "\t-Dhttp.agent=ABCDEF"),
            std::make_pair("\\f-Dhttp.agent=ABCDEF", "\f-Dhttp.agent=ABCDEF"),
            std::make_pair("\\v-Dhttp.agent=ABCDEF", "\v-Dhttp.agent=ABCDEF")
        };

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesEscapeSequenceTest,
    ::testing::ValuesIn(escape_sequences)
);

class ScriptOptionLinesRestTest : public ::testing::TestWithParam<std::string> {};

TEST_P(ScriptOptionLinesRestTest, test_rest_with_tokens) {
    const std::string rest = GetParam();
    /**
    Verify that the parser correctly ignores character sequences containing special parser tokens
    after the options in a line.
    */
    const std::string code =
        "%jvmoption -Dhttp.agent=abc; class Abc{};" + rest;

    options_map_t result;
    parseOptions(code, result);
    ASSERT_EQ(result.size(), 1);

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_NE(jvm_option_result, result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    ASSERT_EQ(jvm_option_result->second[0], buildOption("-Dhttp.agent=abc", 0, 28));
}

const std::vector<std::string> rest_strings =
        {"\\n", "\\r", "something %blabla;", ";", "\\;", "\\;blabla", "\\   blabla", "\\t blabla"};

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesRestTest,
    ::testing::ValuesIn(rest_strings)
);