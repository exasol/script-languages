
#include "include/gtest/gtest.h"
#include "gmock/gmock.h"
#include "base/javacontainer/script_options/string_ops.h"

using namespace SWIGVMContainers::JavaScriptOptions;




TEST(StringOpsTest, trim) {
    std::string sample = " \tHello World \t";
    StringOps::trim(sample);
    EXPECT_EQ(sample, "Hello World");
}

TEST(StringOpsTest, trimWithNoneASCII) {
    /*
    Test that trim works correctly with None-ASCII characters
    \xa0's bit sequence is '1010 0000', while space bit sequence '0010 0000'.
    If StringOps::trim() would not work correctly with characters where MSB is set, it would interpret \xa0 as space.
    */
    std::string sample = " \t\xa0Hello World\xa0 \t";
    StringOps::trim(sample);
    EXPECT_EQ(sample, "\xa0Hello World\xa0");
}

