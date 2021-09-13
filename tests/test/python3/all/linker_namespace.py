#!/usr/bin/env python2.7

import os
import sys


sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf


class LinkerNamespaceTest(udf.TestCase):

    def setUp(self):
        self.query('CREATE SCHEMA FN2', ignore_errors=True)
        self.query('OPEN SCHEMA FN2')

    def test_linker_namespace(self):
        self.query(udf.fixindent('''
                    CREATE OR REPLACE python3 SCALAR SCRIPT linker_namespace_test(search_string VARCHAR(100000)) RETURNS VARCHAR(100000) AS
                    from ctypes import *
                    from typing import List
                    
                    #Wrapper class for accessing libc function dl_iterate_phdr 
                    #which is used to get all shared libraries in link-map (linker namespace)
                    #See https://linux.die.net/man/3/dl_iterate_phdr for details of this function
                    class SymbolScanner:
                    
                        def __init__(self):
                            self.__result = []
                            self.__filter_strings = []
                            #ctypes provides declaration of function pointers, and functions can be implemented in Python. Nice!!!
                            #We declare the interface of the callback type here.
                            self.__callback_t = CFUNCTYPE(c_int,
                                                   POINTER(self.dl_phdr_info),
                                                   POINTER(c_size_t), c_char_p)
                    
                            #Here we actually load the function address to dl_iterate_phdr dynamically
                            self.__dl_iterate_phdr = CDLL('libc.so.6').dl_iterate_phdr
                            # I changed c_void_p to c_char_p
                            # Declare the expected arguments/return types
                            self.__dl_iterate_phdr.argtypes = [self.__callback_t, c_char_p]
                            self.__dl_iterate_phdr.restype = c_int
                    
                    
                        #Declare the struct of dl_phdr_info, also see https://linux.die.net/man/3/dl_iterate_phdr
                        #We are interested only in dlpi_name, so we ignore all other members of the struct
                        class dl_phdr_info(Structure):
                          _fields_ = [
                            ('padding0', c_void_p), # ignore it
                            ('dlpi_name', c_char_p),
                                                    # ignore the reset
                          ]

                        #This is the callback, which dl_iterate_phdr will call for each shared library
                        #We simply check if dlpi_name matches any of our search strings 
                        #(if any search strings is contained in dlpi_name)              
                        def callback(self, info, size, data):
                          # simple search
                            for filter_string in self.__filter_strings:
                                dlpi_name = info.contents.dlpi_name.decode('utf-8')
                                if filter_string in dlpi_name:
                                    self.__result.append(dlpi_name)
                            return 0
                    
                        #Interface of the wrapper. It accepts a list of search strings and returns any shared libraries
                        #which are in the current link-map (linker namespace) which contain any of the search strings
                        def findSymbols(self, filter_strings : List[str]) -> List[str]:
                            self.__filter_strings = filter_strings
                            self.__result = []
                            self.__dl_iterate_phdr(self.__callback_t(self.callback), None)
                            return self.__result

                    def run(ctx):
                        search_string = ctx.search_string;
                        symbol_scanner = SymbolScanner()
                        return ";".join(symbol_scanner.findSymbols([search_string]))
                    '''))
        rows = self.query("SELECT linker_namespace_test(search_string) FROM "
                          "(SELECT search_string FROM (VALUES('proto'), ('zmq')) AS t(search_string))")
        self.assertGreater(len(rows), 0)
        for item in rows:
            self.assertEqual(None, item[0])


if __name__ == '__main__':
    udf.main()
