cmake_minimum_required(VERSION 3.20)
project(Test)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBSYMBOL_SCANNER_HEADERS libSymbolScanner/symbol_scanner.h)
set(LIBSYMBOL_SCANNER_SRCS libSymbolScanner/symbol_scanner.cc)
add_library (symbolscanner SHARED ${LIBSYMBOL_SCANNER_SRCS})
target_link_libraries(symbolscanner dl)

find_package(Protobuf REQUIRED)

add_executable(Test main.cpp)
include_directories(${Protobuf_INCLUDE_DIRS})
message(STATUS "dir='${Protobuf_INCLUDE_DIRS}'")

target_link_libraries(Test PUBLIC dl symbolscanner ${PROTOBUF_LIBRARY})
