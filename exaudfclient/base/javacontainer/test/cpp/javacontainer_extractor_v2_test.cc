
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/test/cpp/javavm_test.h"
#include "base/javacontainer/test/cpp/swig_factory_test.h"
#include "base/javacontainer/javacontainer.h"
#include <string.h>
#include <memory>

using ::testing::MatchesRegex;

class JavaContainerEscapeSequenceTest : public ::testing::TestWithParam<std::pair<std::string, std::string>> {};

TEST_P(JavaContainerEscapeSequenceTest, quoted_jvm_option) {
const std::pair<std::string, std::string> option_value = GetParam();
    const std::string script_code =
        "%jvmoption " + option_value.first + ";\n\n"
        "class JVMOPTION_TEST_WITH_SPACE {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code =
        "package com.exasol;\r\n\n\n"
        "class JVMOPTION_TEST_WITH_SPACE {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(expected_script_code, vm.getJavaVMInternalStatus().m_scriptCode);
    EXPECT_EQ("/exaudf/base/javacontainer/exaudf_deploy.jar", vm.getJavaVMInternalStatus().m_exaJarPath);
    EXPECT_EQ("/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar", vm.getJavaVMInternalStatus().m_classpath);
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);

    const std::vector<std::string> expectedJVMOptions = {   option_value.second, "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(expectedJVMOptions, vm.getJavaVMInternalStatus().m_jvmOptions);
}

const std::vector<std::pair<std::string, std::string>> escape_sequences =
        {
            std::make_pair("-Dhttp.agent=ABC\\nDEF", "-Dhttp.agent=ABC\nDEF"),
            std::make_pair("-Dhttp.agent=ABC\\rDEF", "-Dhttp.agent=ABC\rDEF"),
            std::make_pair("-Dhttp.agent=ABC\\;DEF", "-Dhttp.agent=ABC;DEF"),
            std::make_pair("-Dhttp.agent=ABC\\aDEF", "-Dhttp.agent=ABC\\aDEF"), //any other escape sequence must stay as is
            std::make_pair("\\n-Dhttp.agent=ABCDEF", "\n-Dhttp.agent=ABCDEF"),
            std::make_pair("\\r-Dhttp.agent=ABCDEF", "\r-Dhttp.agent=ABCDEF"),
            std::make_pair("\\;-Dhttp.agent=ABCDEF", ";-Dhttp.agent=ABCDEF"),
            std::make_pair("-Dhttp.agent=ABCDEF\\n", "-Dhttp.agent=ABCDEF\n"),
            std::make_pair("-Dhttp.agent=ABCDEF\\r", "-Dhttp.agent=ABCDEF\r"),
            std::make_pair("-Dhttp.agent=ABCDEF\\;", "-Dhttp.agent=ABCDEF;"),
            std::make_pair("\\ -Dhttp.agent=ABCDEF", "-Dhttp.agent=ABCDEF"),
            std::make_pair("\\t-Dhttp.agent=ABCDEF", "-Dhttp.agent=ABCDEF"),
            std::make_pair("\\f-Dhttp.agent=ABCDEF", "-Dhttp.agent=ABCDEF"),
            std::make_pair("\\v-Dhttp.agent=ABCDEF", "-Dhttp.agent=ABCDEF")
        };

INSTANTIATE_TEST_SUITE_P(
    JavaContainer,
    JavaContainerEscapeSequenceTest,
    ::testing::ValuesIn(escape_sequences)
);

TEST(JavaContainer, import_script_with_escaped_options) {
    const std::string script_code =
        "%import other_script;\n\n"
        "%jvmoption -Dsomeoption=\"ABC\";\n\n"
        "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
        "%jar base/javacontainer/test/test.jar;"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    std::unique_ptr<SwigFactoryTestImpl> swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_script_code =
        "%jvmoption -Dsomeotheroption=\"DE\\nF\";\n\n"
        "%jar base/javacontainer/test/other_test.jar;"
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    swigFactory->addModule("other_script", other_script_code);
    JavaVMTest vm(script_code, std::move(swigFactory));
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code =
        "package com.exasol;\r\n\n\n"
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/other_test.jar:base/javacontainer/test/test.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {    "-Dexasol.scriptclass=com.exasol.udf_profiling.UdfProfiler",
                                                             "-Dsomeotheroption=\"DE\nF\"", "-Dsomeoption=\"ABC\"", "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/other_test.jar:base/javacontainer/test/test.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, basic_jar_with_trailing_escape) {
    const std::string script_code = "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
                                    "%jar base/javacontainer/test/test.jar\\t\t;";
    EXPECT_THROW({
        try
        {
            JavaVMTest vm(script_code);
        }
        catch( const SWIGVMContainers::JavaVMach::exception& e )
        {
            EXPECT_THAT( e.what(), MatchesRegex("^.*Java VM cannot find 'base/javacontainer/test/test\\.jar\t': No such file or directory$"));
            throw;
        }
    }, SWIGVMContainers::JavaVMach::exception );
}
