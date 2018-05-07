package com.exasol;

/**
 * This class is not part of the public Java UDF API
 */
public class ExaUndefinedSingleCallException extends Exception {
    private static final long serialVersionUID = 1L;
    public ExaUndefinedSingleCallException(String undefinedRemoteFn) {
        super("Undefined single call fn: "+undefinedRemoteFn);
        this.undefinedRemoteFn = undefinedRemoteFn;
    }
    public String getUndefinedRemoteFn() {
        return undefinedRemoteFn;
    }
    private String undefinedRemoteFn;
}
