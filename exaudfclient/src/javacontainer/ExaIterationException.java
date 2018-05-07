package com.exasol;

/**
 * This exception indicates an error during interaction with the ExaIterator interface.
 */
public class ExaIterationException extends Exception {
    private static final long serialVersionUID = 1L;
    public ExaIterationException() { super(); }
    public ExaIterationException(String message) { super(message); }
    public ExaIterationException(String message, Throwable cause) { super(message, cause); }
    public ExaIterationException(Throwable cause) { super(cause); }
}
