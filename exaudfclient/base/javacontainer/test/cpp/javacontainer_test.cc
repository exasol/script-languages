
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "javacontainer/test/cpp/javavm_test.h"
#include <string.h>

TEST(JavaContainer, basic_jar) {
    const std::string script_code = "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
                                    "%jar javacontainer/test/test.jar;";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, "\n");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/javacontainer/libexaudf.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/exaudf/javacontainer/libexaudf.jar:javacontainer/test/test.jar");
    const std::vector<std::string> expectedJarPaths = {"javacontainer/test/test.jar"};
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jarPaths, expectedJarPaths);
    EXPECT_FALSE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = { "-Dexasol.scriptclass=com.exasol.udf_profiling.UdfProfiler",
                                                            "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/exaudf/javacontainer/libexaudf.jar:javacontainer/test/test.jar",
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
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code = std::string("package com.exasol;\r\n") + script_code;
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/javacontainer/libexaudf.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/javacontainer/libexaudf.jar");
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_jarPaths.empty());
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/javacontainer/libexaudf.jar",
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
    const std::string script_code_with_jar = std::string("%jar javacontainer/test/test.jar;") + script_code;
    JavaVMTest vm(script_code_with_jar);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    const std::string expected_script_code = std::string("package com.exasol;\r\n") + script_code;
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, expected_script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/javacontainer/libexaudf.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/tmp:/exaudf/javacontainer/libexaudf.jar:javacontainer/test/test.jar");
    const std::vector<std::string> expectedJarPaths = {"javacontainer/test/test.jar"};
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jarPaths, expectedJarPaths);
    EXPECT_TRUE(vm.getJavaVMInternalStatus().m_needsCompilation);
    const std::vector<std::string> expectedJVMOptions = {   "-Xms128m", "-Xmx128m", "-Xss512k",
                                                            "-XX:ErrorFile=/tmp/hs_err_pid%p.log",
                                                            "-Djava.class.path=/tmp:/exaudf/javacontainer/libexaudf.jar:javacontainer/test/test.jar",
                                                            "-XX:+UseSerialGC" };
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_jvmOptions, expectedJVMOptions);
}



