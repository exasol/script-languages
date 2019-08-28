package com.exasol;

import java.math.BigInteger;

/**
 * This interface enables scripts to access metadata such as information about the
 * database and the script. Furthermore it provides related methods, e.g. to
 * access connections and to import scripts.
 */
public interface ExaMetadata {

    /**
     * @return name of the database
     */
    public String getDatabaseName();

    /**
     * @return version of the database
     */
    public String getDatabaseVersion();

    /**
     * @return name and version of the script language, e.g. "Python 2.7.11"
     */
    public String getScriptLanguage();

    /**
     * @return name of the script
     */
    public String getScriptName();

    /**
     * @return name of the script schema
     */
    public String getScriptSchema();

    /**
     * @return name of the script user
     */
    public String getCurrentUser();

    /**
     * @return name of the script scope user
     */
    public String getScopeUser();


    /**
     * @return name of the current open schema
     */
    public String getCurrentSchema();

    /**
     * @return text of the script
     */
    public String getScriptCode();

    /**
     * @return ID of the session in which the current statement is executed
     */
    public String getSessionId();

    /**
     * @return ID of the current statement
     */
    public long getStatementId();

    /**
     * @return number of nodes in the cluster
     */
    public long getNodeCount();

    /**
     * @return ID of the node on which the current JVM is executed, starting with 0
     */
    public long getNodeId();

    /**
     * @return unique ID of the local JVM (the IDs of the virtual machines have no relation to each other)
     */
    public String getVmId();

    /**
     * @return Memory limit for the current JVM process in bytes. If this memory
     * is exceeded, the database resource management will kill the JVM process.
     */
    public BigInteger getMemoryLimit();

    /**
     * @return input type of the script, either "SCALAR" or "SET"
     */
    public String getInputType();

    /**
     * @return number of input columns
     */
    public long getInputColumnCount();

    /**
     * @param column id of the column, starting with 0
     * @return name of the specified input column
     */
    public String getInputColumnName(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return the java class used to represent data from the specified column.
     *         E.g. if this returns java.lang.String, you can access the
     *         data from this column using ExaIterator.getString(column).
     */
    public Class<?> getInputColumnType(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return sql type of the specified column.
     *         This is a string in SQL syntax, e.g. "DECIMAL(18,0)".
     */
    public String getInputColumnSqlType(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return data type precision of the specified column, e.g. the precision of a DECIMAL data type
     */
    public long getInputColumnPrecision(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return data type scale of the specified column, e.g. the scale of a DECIMAL data type
     */
    public long getInputColumnScale(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return data type length of the specified column, e.g. the length of a VARCHAR data type.
     */
    public long getInputColumnLength(int column) throws ExaIterationException;

    /**
     * @return output type of the script, either "RETURNS" or "EMITS"
     */
    public String getOutputType();

    /**
     * @return number of output columns
     */
    public long getOutputColumnCount();

    /**
     * @param column id of the column, starting with 0
     * @return name of the specified output column
     */
    public String getOutputColumnName(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return the java class used to represent data from the specified output column.
     *         E.g. if this returns java.lang.String, you can emit data for this column
     *         using a String.
     */
    public Class<?> getOutputColumnType(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return sql type of the specified output column.
     *         This is a string in SQL syntax, e.g. "DECIMAL(18,0)"
     */
    public String getOutputColumnSqlType(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return data type precision of the specified output column, e.g.
     *         the precision of a DECIMAL data type
     */
    public long getOutputColumnPrecision(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return data type scale of the specified output column, e.g.
     *         the scale of a DECIMAL data type
     */
    public long getOutputColumnScale(int column) throws ExaIterationException;

    /**
     * @param column id of the column, starting with 0
     * @return data type length of the specified column, e.g. the length of a VARCHAR data type
     */
    public long getOutputColumnLength(int column) throws ExaIterationException;

    /**
     * Dynamically loads the code from the specified script, compiles it, and
     * returns an instance of the main script class. You can use Java Reflection
     * to work with this class.
     *
     * Please note that there is a simple way to include other code using
     * the keywords %import and %jar in the script code (see user manual).
     *
     * @param name The name of the script to be imported (context-sensitive)
     * @return instance of the main script class of the imported script
     */
    public Class<?> importScript(String name) throws ExaCompilationException, ClassNotFoundException;

    /**
     * Access the information of an connection (created with CREATE CONNECTION).
     * The executing user must have the according privileges to access the script
     * (see user manual).
     *
     * @param name name of the connection
     * @return an ExaConnectionInformation instance holding all information of the connection
     */
    public ExaConnectionInformation getConnection(String name) throws ExaConnectionAccessException;

}
