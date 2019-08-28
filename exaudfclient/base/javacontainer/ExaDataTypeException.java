package com.exasol;

/**
 * This Exception indicates that a data type error occurred with the script input or output.
 */
public class ExaDataTypeException extends Exception {
    private static final long serialVersionUID = 1L;
    public ExaDataTypeException() { super(); }
    public ExaDataTypeException(String message) { super(message); }
    public ExaDataTypeException(String message, Throwable cause) { super(message, cause); }
    public ExaDataTypeException(Throwable cause) { super(cause); }
}
