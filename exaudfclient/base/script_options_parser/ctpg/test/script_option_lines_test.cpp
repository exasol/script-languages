#include "base/script_options_parser/ctpg/script_option_lines_ctpg.h"
#include <gtest/gtest.h>
#include <string>
#include <exception>

const std::string lineEnd = ";";

class TestException : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

void throwException(const char* ex) {
    throw TestException(std::string(ex));
}

using namespace ExecutionGraph::OptionsLineParser::CTPG;


class ScriptOptionLinesWhitespaceTest : public ::testing::TestWithParam<std::tuple<std::string, std::string, std::string, std::string, std::string, std::string>> {};

TEST_P(ScriptOptionLinesWhitespaceTest, WhitespaceExtractOptionLineTest) {
    const std::string prefix = std::get<0>(GetParam());
    const std::string suffix = std::get<1>(GetParam());
    const std::string option = std::get<2>(GetParam());
    const std::string delimeter = std::get<3>(GetParam());
    const std::string value = std::get<4>(GetParam());
    const std::string payload =  std::get<5>(GetParam());
    std::string code = prefix + "%" + option + delimeter + value + lineEnd + suffix + "\n" + payload;
    options_map_t result;
    parseOptions(code, result, throwException);
    ASSERT_EQ(result.size(), 1);
    const auto option_result = result.find(option);
    ASSERT_TRUE(option_result != result.end());
    ASSERT_EQ(option_result->second.size(), 1);
    EXPECT_EQ(option_result->second[0].value, value);
}

std::vector<std::string> white_space_strings = {"", " ", "\t", "\f", "\v", "\n", " \t", "\t ", "\t\f", "\f\t", "\f ", " \f", "\t\v", "\v\t", "\v ", " \v", "\f\v", "\v\f", "  \t", " \t "};
std::vector<std::string> delimeters = {" ", "\t", "\f", "\v", " \t", "\t ", "\t\f", "\f\t", "\f ", " \f", "\t\v", "\v\t", "\v ", " \v", "\f\v", "\v\f", "  \t", " \t "};
std::vector<std::string> keywords = {"import", "jvmoption", "scriptclass", "jar", "env"};
std::vector<std::string> values = {"something", "com.mycompany.MyScriptClass", "LD_LIBRARY_PATH=/nvdriver", "-Xms128m -Xmx1024m -Xss512k", "/buckets/bfsdefault/default/my_code.jar"};
std::vector<std::string> payloads = {"anything", "\n\ndef my_func:\n\tpass", "class MyJava\n public static void Main() {\n};\n"};

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesWhitespaceTest,
    ::testing::Combine(::testing::ValuesIn(white_space_strings),
                       ::testing::ValuesIn(white_space_strings),
                       ::testing::ValuesIn(keywords),
                       ::testing::ValuesIn(delimeters),
                       ::testing::ValuesIn(values),
                       ::testing::ValuesIn(payloads)
    )
);

TEST(ScriptOptionLinesTest, ignore_anything_other_than_whitepsace) {
    std::string code =
        "abc %option myoption;\n"
        "\nmycode";
    options_map_t result;
    parseOptions(code, result, throwException);
    EXPECT_TRUE(result.empty());
}

TEST(ScriptOptionLinesTest, need_line_end_character) {
    std::string code =
        "%option myoption\n"
        "\nmycode";
    options_map_t result;
    EXPECT_THROW({
       parseOptions(code, result, throwException);
    }, TestException );
}

TEST(ScriptOptionLinesTest, finds_the_two_options_same_key) {
    std::string code =
        "%option myoption; %option mysecondoption;\n"
        "\nmycode";
    options_map_t result;
    parseOptions(code, result, throwException);
    ASSERT_EQ(result.size(), 1);
    const auto option_result = result.find("option");

    ASSERT_TRUE(option_result != result.end());
    ASSERT_EQ(option_result->second.size(), 2);
    ScriptOption expected_option_one = { .value = "myoption", .idx_in_source = 0, .size = 17};
    ScriptOption expected_option_two = { .value = "mysecondoption", .idx_in_source = 18, .size = 23};
    ASSERT_EQ(option_result->second[0], expected_option_one);
    ASSERT_EQ(option_result->second[1], expected_option_two);
    EXPECT_EQ(code.substr(expected_option_one.idx_in_source, expected_option_one.size), "%option myoption;");
    EXPECT_EQ(code.substr(expected_option_two.idx_in_source, expected_option_two.size), "%option mysecondoption;");
}

TEST(ScriptOptionLinesTest, finds_the_two_options_different_keys) {
    std::string code =
        "%option myoption; %otheroption mysecondoption;\n"
        "\nmycode";
    options_map_t result;
    parseOptions(code, result, throwException);
    ASSERT_EQ(result.size(), 2);
    const auto option_result = result.find("option");

    ASSERT_TRUE(option_result != result.end());
    ASSERT_EQ(option_result->second.size(), 1);
    ScriptOption expected_option_one = { .value = "myoption", .idx_in_source = 0, .size = 17};
    ASSERT_EQ(option_result->second[0], expected_option_one);
    EXPECT_EQ(code.substr(expected_option_one.idx_in_source, expected_option_one.size), "%option myoption;");

    const auto otheroption_result = result.find("otheroption");

    ASSERT_TRUE(otheroption_result != result.end());
    ASSERT_EQ(otheroption_result->second.size(), 1);
    ScriptOption expected_option_two = { .value = "mysecondoption", .idx_in_source = 18, .size = 28};
    ASSERT_EQ(otheroption_result->second[0], expected_option_two);
    EXPECT_EQ(code.substr(expected_option_two.idx_in_source, expected_option_two.size), "%otheroption mysecondoption;");
}

class ScriptOptionLinesInvalidOptionTest : public ::testing::TestWithParam<std::string> {};


TEST_P(ScriptOptionLinesInvalidOptionTest, value_is_mandatory) {
    const std::string invalid_option = GetParam();
    std::string code = invalid_option + "\nsomething";
    options_map_t result;
    EXPECT_THROW({
     parseOptions(code, result, throwException);
    }, TestException );
}

std::vector<std::string> invalid_options = {"%option ;", "%option \n", "\n%option\n;", "%option\nvalue;"};

INSTANTIATE_TEST_SUITE_P(
    ScriptOptionLines,
    ScriptOptionLinesInvalidOptionTest,
    ::testing::ValuesIn(invalid_options)
);



TEST(ScriptOptionLinesTest, test_all_in_one_line_does_second_option_does_not_work) {
    /**
    Verify the correct behavior of new parser for situation as described in https://github.com/exasol/script-languages-release/issues/652.
    */
    const std::string code = "%jar /buckets/bucketfs1/jars/exajdbc.jar; %jvmoption -Xms4m; class JAVA_UDF_3 {static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {String host_name = ctx.getString(\"col1\");}}\n/\n;";
    options_map_t result;
    parseOptions(code, result, throwException);
    ASSERT_EQ(result.size(), 2);

    const auto jar_option_result = result.find("jar");
    ASSERT_TRUE(jar_option_result != result.end());
    ASSERT_EQ(jar_option_result->second.size(), 1);
    ScriptOption expected_jar_option = { .value = "/buckets/bucketfs1/jars/exajdbc.jar", .idx_in_source = 0, .size = 41};
    ASSERT_EQ(jar_option_result->second[0], expected_jar_option);
    EXPECT_EQ(code.substr(expected_jar_option.idx_in_source, expected_jar_option.size), "%jar /buckets/bucketfs1/jars/exajdbc.jar;");

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_TRUE(jvm_option_result != result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    ScriptOption expected_jvm_option = { .value = "-Xms4m", .idx_in_source = 42, .size = 18};
    ASSERT_EQ(jvm_option_result->second[0], expected_jvm_option);
    EXPECT_EQ(code.substr(expected_jvm_option.idx_in_source, expected_jvm_option.size), "%jvmoption -Xms4m;");
}


TEST(ScriptOptionLinesTest, test_values_must_not_contain_spaces) {
    /**
    Verify the wrong behavior and assumptions as described in https://github.com/exasol/script-languages-release/issues/878
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
    parseOptions(code, result, throwException);
    ASSERT_EQ(result.size(), 1);

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_TRUE(jvm_option_result != result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    ScriptOption expected_jvm_option = { .value = "-Dhttp.agent=\"ABC DEF\"", .idx_in_source = 0, .size = 34};
    ASSERT_EQ(jvm_option_result->second[0], expected_jvm_option);
    EXPECT_EQ(code.substr(expected_jvm_option.idx_in_source, expected_jvm_option.size), "%jvmoption -Dhttp.agent=\"ABC DEF\";");
}

TEST(ScriptOptionLinesTest, test_multiple_lines_with_code) {
    /**
    Verify that the parser can read options coming after some code.
    */
    const std::string code =
        "%jvmoption -Dhttp.agent=\"ABC DEF\"; class Abc{};\n\n"
        "%jar /buckets/bucketfs1/jars/exajdbc.jar; class DEF{};\n";

    options_map_t result;
    parseOptions(code, result, throwException);
    ASSERT_EQ(result.size(), 2);

    const auto jvm_option_result = result.find("jvmoption");
    ASSERT_TRUE(jvm_option_result != result.end());
    ASSERT_EQ(jvm_option_result->second.size(), 1);
    ScriptOption expected_jvm_option = { .value = "-Dhttp.agent=\"ABC DEF\"", .idx_in_source = 0, .size = 34};
    ASSERT_EQ(jvm_option_result->second[0], expected_jvm_option);
    EXPECT_EQ(code.substr(expected_jvm_option.idx_in_source, expected_jvm_option.size), "%jvmoption -Dhttp.agent=\"ABC DEF\";");

    const auto jar_option_result = result.find("jar");
    ASSERT_TRUE(jar_option_result != result.end());
    ASSERT_EQ(jar_option_result->second.size(), 1);
    ScriptOption expected_jar_option = { .value = "/buckets/bucketfs1/jars/exajdbc.jar", .idx_in_source = 49, .size = 41};
    ASSERT_EQ(jar_option_result->second[0], expected_jar_option);
    EXPECT_EQ(code.substr(expected_jar_option.idx_in_source, expected_jar_option.size), "%jar /buckets/bucketfs1/jars/exajdbc.jar;");
}
