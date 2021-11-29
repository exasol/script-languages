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