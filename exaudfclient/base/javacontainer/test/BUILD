java_library(
    name = "ExaStackTraceCleaner",
    srcs = ["//base/javacontainer:ExaStackTraceCleaner.java"]
)

java_test(
    name = "ExaStackTraceCleanerTest",
    srcs = ["java/com/exasol/ExaStackTraceCleanerTest.java"],
    size = "small",
    deps = [":ExaStackTraceCleaner"]
)


cc_test(
    name = "javacontainer-test",
    srcs = ["cpp/javacontainer_test.cc", "cpp/exaudf_wrapper.cc", "cpp/javavm_test.cc", "cpp/javavm_test.h",
            "cpp/swig_factory_test.h", "cpp/swig_factory_test.cc"],
    deps = [
        "//base/javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
    data = ["test.jar", "other_test.jar"]
)