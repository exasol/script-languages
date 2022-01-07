
#include "include/gtest/gtest.h"
#include "javacontainer/test/cpp/javavm_test.h"


TEST(JavaContainer, basic_inline) {
    const std::string script_code = "%scriptclass com.exasol.udf_profiling.UdfProfiler;\n"
                                    "%jar /buckets/bfsdefault/myudfs/ProfilingUdf-1.0-SNAPSHOT.jar;";
    JavaVMTest vm(script_code);
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJavaPath, "/exaudf/javacontainer");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_localClasspath, "");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_scriptCode, "");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_exaJarPath, "");
    EXPECT_EQ(vm.getJavaVMInternalStatus().m_classpath, "");

}