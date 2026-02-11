java_library(
    name = "ExaStackTraceCleaner",
    srcs = ["//javacontainer:ExaStackTraceCleaner.java"]
)

java_test(
    name = "ExaStackTraceCleanerTest",
    srcs = ["java/com/exasol/ExaStackTraceCleanerTest.java"],
    size = "small",
    deps = [":ExaStackTraceCleaner"]
)

JAVACONTAINER_TEST_SRCS = ["cpp/javacontainer_test.cc", "cpp/exaudf_wrapper.cc", "cpp/javavm_test.cc", "cpp/javavm_test.h",
            "cpp/swig_factory_test.h", "cpp/swig_factory_test.cc"]

JAVACONTAINER_PERF_TEST_SRCS = ["cpp/javacontainer_perf_test.cc", "cpp/exaudf_wrapper.cc", "cpp/javavm_test.cc", "cpp/javavm_test.h",
            "cpp/swig_factory_test.h", "cpp/swig_factory_test.cc"]

cc_test(
    name = "javacontainer-test-extractor-legacy",
    srcs = JAVACONTAINER_TEST_SRCS,
    deps = [
        "//javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
    data = ["test.jar", "other_test.jar"]
)

cc_test(
    name = "javacontainer-test-extractor-v2",
    srcs = JAVACONTAINER_TEST_SRCS + ["cpp/javacontainer_extractor_v2_test.cc"],
    deps = [
        "//javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
    defines = ["USE_EXTRACTOR_V2"],
    data = ["test.jar", "other_test.jar"]
)


cc_test(
    name = "javacontainer-perf-test-legacy-parser",
    srcs = JAVACONTAINER_PERF_TEST_SRCS,
    deps = [
        "//javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
    data = ["test.jar", "other_test.jar"]
)

cc_test(
    name = "javacontainer-perf-test-ctpg-parser",
    srcs = JAVACONTAINER_PERF_TEST_SRCS,
    deps = [
        "//javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
    defines = ["USE_EXTRACTOR_V2"],
    data = ["test.jar", "other_test.jar"]
)
