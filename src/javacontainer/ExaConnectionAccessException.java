package com.exasol;

/**
 * This exception indicates that the requested connection information could not
 * be accessed, e.g. because the user has insufficient privileges or the connection
 * does not exist.
 */
public class ExaConnectionAccessException extends Exception {
    private static final long serialVersionUID = 1L;
    public ExaConnectionAccessException(String msg) {
        super("connection exception: "+msg);
        this.msg = msg;
    }
    public String getMessage() {
        return msg;
    }
    private String msg;
}
