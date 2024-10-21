
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/test/cpp/javavm_test.h"
#include "base/javacontainer/javacontainer.h"
#include "base/javacontainer/test/cpp/swig_factory_test.h"
#include <string.h>

using ::testing::MatchesRegex;


TEST(JavaContainer, basic_jar) {
    const std::string script_code = "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
                                    "%jar base/javacontainer/test/test.jar;";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, "\n");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar");
    EXPECT_FALSE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = { "-Dexasol.scriptclass=com.exasol.udf_profiling.UdfProfiler",
                                                            "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, basic_jar_script_class_with_white_spaces) {
    const std::string script_code = "%scriptclass com.exasol.udf_profiling.UdfProfiler\t    ;\n"
                                    "%jar base/javacontainer/test/test.jar;";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, "\n");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar");
    EXPECT_FALSE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = { "-Dexasol.scriptclass=com.exasol.udf_profiling.UdfProfiler",
                                                            "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}


TEST(JavaContainer, basic_jar_with_white_spaces) {
    const std::string script_code = "%jar base/javacontainer/test/test.jar \t ;";

#ifndef USE_CTPG_PARSER //The parsers behave differently: The legacy parser removes trailing white spaces.
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar");
#else
    EXPECT_THROW({
        try
        {
            JavaVMTest vm(script_code);
        }
        catch( const SWIGVMContainers::JavaVMach::exception& e )
        {
            EXPECT_THAT( e.what(), MatchesRegex("^.*Java VM cannot find 'base/javacontainer/test/test\\.jar \t ': No such file or directory$"));
            throw;
        }
    }, SWIGVMContainers::JavaVMach::exception );
#endif
}


TEST(JavaContainer, basic_inline) {
    const std::string script_code = "import java.time.LocalDateTime;"
                                    "import java.time.ZoneOffset;"
                                    "import java.time.format.DateTimeFormatter;"
                                    "class SIMPLE {"
                                    "static void main(String[] args) {"
                                    "int i = 0;"
                                    "}"
                                    "static int run(ExaMetadata exa, ExaIterator ctx) {"
                                    "return 0;"
                                    "}"
                                    "}";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code = std::string("package com.exasol;\r\n") + script_code;
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}



TEST(JavaContainer, combined_inline_jar) {
    const std::string script_code = "import java.time.LocalDateTime;"
                                    "import java.time.ZoneOffset;"
                                    "import java.time.format.DateTimeFormatter;"
                                    "class SIMPLE {"
                                    "static void main(String[] args) {"
                                    "int i = 0;"
                                    "}"
                                    "static int run(ExaMetadata exa, ExaIterator ctx) {"
                                    "return 0;"
                                    "}"
                                    "}";
    const std::string script_code_with_jar = std::string("%jar base/javacontainer/test/test.jar;") + script_code;
    JavaVMTest vm(script_code_with_jar);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code = std::string("package com.exasol;\r\n") + script_code;
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}


TEST(JavaContainer, quoted_jvm_option) {
    const std::string script_code =
        "%jvmoption -Dhttp.agent=\"ABC DEF\";\n\n"
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
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    /*
     * Note: The option "DEF" is wrong and causes UDF's to crash!
     *       The correct option would be '-Dhttp.agent=\"ABC DEF\"'
     */
    const std::vector<std::string> expectedJVMOptions = {   "-Dhttp.agent=\"ABC", "DEF\"", "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, simple_import_script) {
    const std::string script_code =
        "%import other_script;\n\n"
        "%jvmoption -Dhttp.agent=\"ABC\";\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_script_code =
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    swigFactory->addModule("other_script", other_script_code);
    JavaVMTest vm(script_code, std::move(swigFactory));
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code =
        "package com.exasol;\r\n"
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Dhttp.agent=\"ABC\"", "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, simple_import_script_with_white_space) {
    const std::string script_code =
        "%import other_script\t ;\n\n"
        "%jvmoption -Dhttp.agent=\"ABC\";\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_script_code =
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    swigFactory->addModule("other_script", other_script_code);
    JavaVMTest vm(script_code, std::move(swigFactory));
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code =
        "package com.exasol;\r\n"
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Dhttp.agent=\"ABC\"", "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, import_script_with_recursion) {
    const std::string script_code =
        "%import other_script;\n\n"
        "%jvmoption -Dhttp.agent=\"ABC\";\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_script_code =
        "%import other_script;\n\n"
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
        "}\n\n\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Dhttp.agent=\"ABC\"", "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, import_script_with_jvmoption) {
    const std::string script_code =
        "%import other_script;\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_script_code =
        "%jvmoption -Dhttp.agent=\"ABC\";\n\n"
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
        "}\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Dhttp.agent=\"ABC\"", "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, multiple_import_scripts) {
    const std::string script_code =
        "%import other_script_A;\n\n"
        "%import other_script_C;\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_scipt_code_A =
        "%import other_script_B;\n\n"
        "class OtherClassA {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    const std::string other_scipt_code_B =
        "class OtherClassB {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    const std::string other_scipt_code_C =
        "%import other_script_B;\n\n"
        "class OtherClassC {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    swigFactory->addModule("other_script_A", other_scipt_code_A);
    swigFactory->addModule("other_script_B", other_scipt_code_B);
    swigFactory->addModule("other_script_C", other_scipt_code_C);
    JavaVMTest vm(script_code, std::move(swigFactory));
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code =
        "package com.exasol;\r\n"
        "class OtherClassB {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n"
        "class OtherClassA {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n\n\n"
        "class OtherClassC {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, import_script_with_mixed_options) {
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
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_script_code =
        "%jvmoption -Dsomeotheroption=\"DEF\";\n\n"
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
                                                             "-Dsomeotheroption=\"DEF\"", "-Dsomeoption=\"ABC\"", "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/other_test.jar:base/javacontainer/test/test.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}

TEST(JavaContainer, import_script_script_class_option_ignored) {
    const std::string script_code =
        "%import other_script;\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_script_code =
        "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    swigFactory->addModule("other_script", other_script_code);
    JavaVMTest vm(script_code, std::move(swigFactory));
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code =
        "package com.exasol;\r\n"
#ifndef USE_CTPG_PARSER //The parsers behave differently: The legacy parser incorrectly keeps imported scriptclass options
        "%scriptclass com.exasol.udf_profiling.UdfProfiler;"
#endif
        "\n"
        "class OtherClass {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {    "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}


TEST(JavaContainer, import_scripts_deep_recursion) {
    const std::string script_code =
        "%import other_script_A;\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "	ctx.emit(\"Success!\");\n"
        " }\n"
        "}\n";
    auto swigFactory = std::make_unique<SwigFactoryTestImpl>();

    const std::string other_scipt_code_A =
        "%import other_script_B;\n\n"
        "class OtherClassA {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    const std::string other_scipt_code_B =
        "%import other_script_C;\n\n"
        "class OtherClassB {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    const std::string other_scipt_code_C =
        "%import other_script_D;\n\n"
        "class OtherClassC {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    const std::string other_scipt_code_D =
        "%import other_script_A;\n\n"
        "class OtherClassD {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n";
    swigFactory->addModule("other_script_A", other_scipt_code_A);
    swigFactory->addModule("other_script_B", other_scipt_code_B);
    swigFactory->addModule("other_script_C", other_scipt_code_C);
    swigFactory->addModule("other_script_D", other_scipt_code_D);
    JavaVMTest vm(script_code, std::move(swigFactory));
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code =
        "package com.exasol;\r\n\n\n"
        "class OtherClassD {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n"
        "class OtherClassC {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n"
        "class OtherClassB {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n"
        "class OtherClassA {\n"
        "static void doSomething() {\n\n"
        " }\n"
        "}\n\n\n"
        "class JVMOPTION_TEST {\n"
        "static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {\n\n"
        "\tctx.emit(\"Success!\");\n"
         " }\n}\n";
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/base/javacontainer/exaudf_deploy.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}
