package com.exasol;

/**
 * This exception indicates that the compilation of a script failed, or
 * that the script api callback functions are not correctly implemented.
 */
public class ExaCompilationException extends Exception {
    private static final long serialVersionUID = 1L;
    public ExaCompilationException() { super(); }
    public ExaCompilationException(String message) { super(message); }
    public ExaCompilationException(String message, Throwable cause) { super(message, cause); }
    public ExaCompilationException(Throwable cause) { super(cause); }
}
