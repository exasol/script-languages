#!/usr/bin/env python2.7

import os
import string
from typing import List
import sys
from ctypes import *

# libProto = CDLL("libprotobuf.so", mode = RTLD_GLOBAL)
# lib = CDLL("/home/thomas/Work/script-languages-release/script-languages/tests/test/generic/symbol-scanner/binary/libsymbolscanner.so")
# lib.scan_symbols.restype = c_char_p
# p = create_string_buffer(b"proto")
# r = lib.scan_symbols(p)
# res = c_char_p(r)
# t = res.value.decode("utf-8")
# print(f'Found:{t}')
# #res = c_int(lib.my_func(5))
# #file = open("tst2.log", 'w')



class SymbolScanner:

    def __init__(self):
        self.__result = []
        self.__filter_strings = []
        self.__callback_t = CFUNCTYPE(c_int,
                               POINTER(self.dl_phdr_info),
                               POINTER(c_size_t), c_char_p)

        self.__dl_iterate_phdr = CDLL('libc.so.6').dl_iterate_phdr
        # I changed c_void_p to c_char_p
        self.__dl_iterate_phdr.argtypes = [self.__callback_t, c_char_p]
        self.__dl_iterate_phdr.restype = c_int


    class dl_phdr_info(Structure):
      _fields_ = [
        ('padding0', c_void_p), # ignore it
        ('dlpi_name', c_char_p),
                                # ignore the reset
      ]

    def findSymbols(self, filter_strings : List[str]) -> List[str]:
        self.__filter_strings = filter_strings
        self.__result = []
        self.__dl_iterate_phdr(self.__callback_t(self.callback), None)
        return self.__result

    def callback(self, info, size, data):
      # simple search
        for filter_string in self.__filter_strings:
            dlpi_name = info.contents.dlpi_name.decode('utf-8')
            if filter_string in dlpi_name:
                self.__result.append(dlpi_name)
        return 0

if __name__ == '__main__':
   s = SymbolScanner()
   print(f'{s.findSymbols(["c"])}')

