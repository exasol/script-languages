
#include "include/gtest/gtest.h"
#include ""

void* load_dynamic(const char* name) {
    return nullptr;
}

namespace SWIGVMContainers {
__thread SWIGVM_params_t * SWIGVM_params = nullptr;
}


TEST(JavaContainer, basic_inline) {
    EXPECT_EQ(1+1, 2);
}