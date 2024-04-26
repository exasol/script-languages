package com.exasol.slc.testudf;

import java.lang.Runtime.Version;

import com.exasol.ExaIterator;
import com.exasol.ExaMetadata;


public class Main {

    private static boolean initCalled = false;

    public Main() {
        throw new RuntimeException("UDF client should not call constructor");
    }

    public static void init(ExaMetadata exa) throws Exception {
        System.out.println("Init called");
        initCalled = true;
    }

    public static void cleanup(ExaMetadata exa) throws Exception {
        System.out.println("Cleanup called");
    }
 
    public static int run(final ExaMetadata exa, final ExaIterator ctx) {
        if(!initCalled) {
            throw new RuntimeException("UDF client did not call init method");
        }
        final Version version = Runtime.version();
        System.out.println("Java version: " + version);
        return version.feature();
    }
}
