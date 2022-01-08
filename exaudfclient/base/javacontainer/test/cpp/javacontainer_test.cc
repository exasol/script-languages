
#include "include/gtest/gtest.h"
#include "javacontainer/test/cpp/javavm_test.h"


TEST(JavaContainer, basic_inline) {
    const std::string script_code = "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
                                    "%jar javacontainer/test/test.jar;";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "/tmp");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, "\n");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "/exaudf/javacontainer/libexaudf.jar");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "/exaudf/javacontainer/libexaudf.jar:javacontainer/test/test.jar");

}