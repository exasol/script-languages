package com.exasol;

import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;

/**
 * This interface enables UDF scripts to iterate over input data and to emit
 * output.
 */
public interface ExaIterator {

    /**
     * @return number of input rows for for this script.
     *         If this is a "SET" UDF script, it will return the number of rows for the current group.
     *         If this is a "SCALAR" UDF script, it will return the total number of rows to be processed by this JVM instance.
     */
    public long size() throws ExaIterationException;

    /**
     * Increases the iterator to the next row of the current group, if there is a new row.
     * It only applies for "SET" UDF scripts. The iterator initially points to the first row,
     * so call this method after processing a row.
     *
     * The following code can be used to process all rows of a group:
     * <blockquote><pre>
     * public static void run(ExaMetadata meta, ExaIterator iter) throws Exception {
     *   do {
     *     // access data here, e.g. with iter.getString("MY_COLUMN");
     *   } while (iter.next());
     * }
     * </pre></blockquote>
     *
     * @return true, if the is a next row and the iterator was increased to it, false,
     *         if there is no more row for this group
     */
    public boolean next() throws ExaIterationException;

    /**
     * Resets the iterator to the first input row. This is only allowed for "SET" UDF scripts.
     */
    public void reset() throws ExaIterationException;

    /**
     * Emit an output row. This is only allowed for "SET" UDF scripts.
     * Note that you can emit using multiple function arguments or an object array:
     * <blockquote><pre>
     * iter.emit(1, "a");
     * iter.emit(new Object[] {1, "a"});
     * </pre></blockquote>
     */
    public void emit(Object... values) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as an Integer object.
     *         This can be used for the data type DECIMAL(p,0).
     *
     * @param column index of the column, starting with 0
     */
    public Integer getInteger(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as an Integer object.
     *         This can be used for the data type DECIMAL(p,0).
     *
     * @param name name of the column
     */
    public Integer getInteger(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a Long object.
     *         This can be used for the data type DECIMAL(p,0).
     *
     * @param column index of the column, starting with 0
     */
    public Long getLong(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a Long object.
     *         This can be used for the data type DECIMAL(p,0).
     *
     * @param name name of the column
     */
    public Long getLong(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a BigDecimal object.
     *         This can be used for the data type DECIMAL(p,0) and DECIMAL(p,s).
     *
     * @param column index of the column, starting with 0
     */
    public BigDecimal getBigDecimal(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a BigDecimal object.
     *         This can be used for the data type DECIMAL(p,0) and DECIMAL(p,s).
     *
     * @param name name of the column
     */
    public BigDecimal getBigDecimal(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a Double object.
     *         This can be used for the data type DOUBLE.
     *
     * @param column index of the column, starting with 0
     */
    public Double getDouble(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a Double object.
     *         This can be used for the data type DOUBLE.
     *
     * @param name name of the column, starting with 0
     */
    public Double getDouble(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a String object.
     *         This can be used for the data type VARCHAR and CHAR.
     *
     * @param column index of the column, starting with 0
     */
    public String getString(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a String object.
     *         This can be used for the data type VARCHAR and CHAR.
     *
     * @param name name of the column
     */
    public String getString(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a Boolean object.
     *         This can be used for the data type BOOLEAN.
     *
     * @param column index of the column, starting with 0
     */
    public Boolean getBoolean(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a Boolean object.
     *         This can be used for the data type BOOLEAN.
     *
     * @param name name of the column
     */
    public Boolean getBoolean(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a {@link java.sql.Date} object.
     *         This can be used for the data type DATE.
     *
     * @param column index of the column, starting with 0
     */
    public Date getDate(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a {@link java.sql.Date} object.
     *         This can be used for the data type DATE.
     *
     * @param name name of the column
     */
    public Date getDate(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a {@link java.sql.Timestamp} object.
     *         This can be used for the data type TIMESTAMP.
     *
     * @param column index of the column, starting with 0
     */
    public Timestamp getTimestamp(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row as a {@link java.sql.Timestamp} object.
     *         This can be used for the data type TIMESTAMP.
     *
     * @param name name of the column
     */
    public Timestamp getTimestamp(String name) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row.
     *         This can be used for all data types. You have to cast the value appropriately.
     *
     * @param column index of the column, starting with 0
     */
    public Object getObject(int column) throws ExaIterationException, ExaDataTypeException;

    /**
     * @return value of the specified column of the current row.
     *         This can be used for all data types. You have to cast the value appropriately.
     *
     * @param name name of the column
     */
    public Object getObject(String name) throws ExaIterationException, ExaDataTypeException;
}
