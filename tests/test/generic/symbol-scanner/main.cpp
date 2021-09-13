#include <iostream>
#include <dlfcn.h>

#include "libSymbolScanner/symbol_scanner.h"

#include <google/protobuf/timestamp.pb.h>
#include <google/protobuf/util/time_util.h>


int main() {
    google::protobuf::Timestamp t;
    std::cout << google::protobuf::util::TimeUtil::ToString(t) << std::endl;


//    void *lib_protobuf = ::dlopen("libprotobuf.so", RTLD_NOW);
//    void *lib_protobuf = ::dlmopen(LM_ID_NEWLM, "libprotobuf.so", RTLD_NOW | RTLD_GLOBAL);

    std::cerr << "Result:" << scan_symbols("protobuf\tc++") << std::endl;
    return 0;
}
