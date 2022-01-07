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


cc_test(
    name = "javacontainer-test",
    srcs = ["cpp/javacontainer_test.cc"],
    deps = [
        "//javacontainer:javacontainer",
        "@googletest//:gtest_main",
    ],
)