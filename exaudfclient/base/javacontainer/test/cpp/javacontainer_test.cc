
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/test/cpp/javavm_test.h"
#include <string.h>

TEST(JavaContainer, basic_jar) {
    const std::string script_code = "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
                                    "%jar base/javacontainer/test/test.jar;";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/base/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, "\n");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/base/javacontainer/exaudf_deploy.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar");
    const std::vector<std::string> expectedJarPaths = {"base/javacontainer/test/test.jar"};
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jarPaths, expectedJarPaths);
    EXPECT_FALSE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = { "-Dexasol.scriptclass=com.exasol.udf_profiling.UdfProfiler",
                                                            "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/exaudf/base/javacontainer/exaudf_deploy.jar:base/javacontainer/test/test.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
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
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_jarPaths.empty());
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
    const std::vector<std::string> expectedJarPaths = {"base/javacontainer/test/test.jar"};
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jarPaths, expectedJarPaths);
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
    const std::vector<std::string> expectedJarPaths = {};
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jarPaths, expectedJarPaths);
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
