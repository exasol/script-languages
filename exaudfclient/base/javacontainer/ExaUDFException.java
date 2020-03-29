package com.exasol;

/**
 * This exception indicates that an exception during the execution of user code happend.
 */
public class ExaUDFException extends Exception {
    private static final long serialVersionUID = 1L;
    public ExaUDFException() { super(); }
    public ExaUDFException(String message) { super(message); }
    public ExaUDFException(String message, Throwable cause) { super(message, cause); }
    public ExaUDFException(Throwable cause) { super(cause); }
}
