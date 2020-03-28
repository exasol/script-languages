package com.exasol;

import java.util.ArrayList;
import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.lang.reflect.InvocationTargetException;
import java.math.BigDecimal;
import java.sql.Date;
import java.sql.Timestamp;
import java.io.IOError;
import com.exasol.swig.ResultHandler;
import com.exasol.swig.TableIterator;
import com.exasol.swig.Metadata;

class ExaIteratorImpl implements ExaIterator {
    private ExaMetadata exaMetadata;
    private TableIterator tableIterator;
    private ResultHandler resultHandler;
    private boolean finished;
    private boolean singleInput;
    private boolean singleOutput;
    private boolean insideRun;
    private Object[] cache;
    private String[] inputColumnTypes;
    private String[] outputColumnTypes;
    private ArrayList<String> columnNames;

    @Override
    public long size() throws ExaIterationException {
        long size = tableIterator.rowsInGroup();
        String exMsg = tableIterator.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("F-UDF.CL.J-57: "+exMsg);
        }
        return size;
    }

    @Override
    public boolean next() throws ExaIterationException {
        if (insideRun && singleInput)
            throw new ExaIterationException("E-UDF.CL.J-58: next() function is not allowed in scalar context");
        clearCache();
        if (finished)
            return false;
        boolean next = tableIterator.next();
        String exMsg = tableIterator.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("F-UDF.CL.J-59: "+exMsg);
        }
        if (!next)
            finished = true;
        return next;
    }

    @Override
    public void reset() throws ExaIterationException {
        if (singleInput)
            throw new ExaIterationException("E-UDF.CL.J-60: reset() function is not allowed in scalar context");
        clearCache();
        tableIterator.reset();
        String exMsg = tableIterator.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("F-UDF.CL.J-61: "+exMsg);
        }
        finished = false;
    }

    @Override
    public void emit(Object... values) throws ExaIterationException, ExaDataTypeException {
        if (insideRun && singleOutput)
            throw new ExaIterationException("E-UDF.CL.J-62: emit() function is not allowed in scalar context");

        if(values != null){
            if (values.length != exaMetadata.getOutputColumnCount()) {
                String errorText = "E-UDF.CL.J-63: emit() takes exactly " + exaMetadata.getOutputColumnCount();
                errorText += (exaMetadata.getOutputColumnCount() > 1) ? " arguments" : " argument";
                errorText += " (" + values.length + " given)";
                throw new ExaIterationException(errorText);
            }

            for (int i = 0; i < values.length; i++) {
                if (values[i] == null) {
                    resultHandler.setNull(i);
                }
                else if (values[i] instanceof Byte || values[i] instanceof Short || values[i] instanceof Integer) {
                    Number val = (Number) values[i];
                    if (outputColumnTypes[i].equals("INT32"))
                        resultHandler.setInt32(i, val.intValue());
                    else if (outputColumnTypes[i].equals("INT64"))
                        resultHandler.setInt64(i, val.longValue());
                    else if (outputColumnTypes[i].equals("NUMERIC"))
                        resultHandler.setNumeric(i, val.toString());
                    else if (outputColumnTypes[i].equals("DOUBLE"))
                        resultHandler.setDouble(i, val.doubleValue());
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-64: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type " + 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else if (values[i] instanceof Long) {
                    Long val = (Long) values[i];
                    if (outputColumnTypes[i].equals("INT32")) {
                        if (isConversionToIntegerSafe(val, exaMetadata.getOutputColumnName(i)))
                            resultHandler.setInt32(i, val.intValue());
                    }
                    else if (outputColumnTypes[i].equals("INT64"))
                        resultHandler.setInt64(i, val.longValue());
                    else if (outputColumnTypes[i].equals("NUMERIC"))
                        resultHandler.setNumeric(i, val.toString());
                    else if (outputColumnTypes[i].equals("DOUBLE"))
                        resultHandler.setDouble(i, val.doubleValue());
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-65: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type " + 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else if (values[i] instanceof Float || values[i] instanceof Double) {
                    Number val = (Number) values[i];
                    if (outputColumnTypes[i].equals("INT32")) {
                        if (isConversionToIntegerSafe(val, exaMetadata.getOutputColumnName(i)))
                            resultHandler.setInt32(i, val.intValue());
                    }
                    else if (outputColumnTypes[i].equals("INT64")) {
                        if (isConversionToLongSafe(val, exaMetadata.getOutputColumnName(i)))
                            resultHandler.setInt64(i, val.longValue());
                    }
                    else if (outputColumnTypes[i].equals("NUMERIC"))
                        resultHandler.setNumeric(i, val.toString());
                    else if (outputColumnTypes[i].equals("DOUBLE"))
                        resultHandler.setDouble(i, val.doubleValue());
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-66: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type "+ 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else if (values[i] instanceof BigDecimal) {
                    BigDecimal val = (BigDecimal) values[i];
                    if (outputColumnTypes[i].equals("INT32"))
                        resultHandler.setInt32(i, val.intValueExact());
                    else if (outputColumnTypes[i].equals("INT64"))
                        resultHandler.setInt64(i, val.longValueExact());
                    else if (outputColumnTypes[i].equals("NUMERIC"))
                        resultHandler.setNumeric(i, val.toString());
                    else if (outputColumnTypes[i].equals("DOUBLE"))
                        resultHandler.setDouble(i, val.doubleValue());
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-67: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type " + 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else if (values[i] instanceof Boolean) {
                    Boolean val = (Boolean) values[i];
                    if (outputColumnTypes[i].equals("BOOLEAN"))
                        resultHandler.setBoolean(i, val.booleanValue());
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-68: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type " + 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else if (values[i] instanceof String) {
                    String val = (String) values[i];
                    if (outputColumnTypes[i].equals("STRING")) {
                        byte[] utf8Bytes = null;
                        try {
                            utf8Bytes = val.getBytes("UTF-8");
                        } catch (java.io.UnsupportedEncodingException ex) {
                            throw new ExaDataTypeException(
                                "E-UDF.CL.J-69: Column with name '" + 
                                exaMetadata.getOutputColumnName(i) + 
                                "' contains invalid UTF-8 data"
                                );
                        }
                        resultHandler.setString(i, utf8Bytes);
                    }
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-70: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type " + 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else if (values[i] instanceof Date) {
                    Date val = (Date) values[i];
                    if (outputColumnTypes[i].equals("DATE"))
                        resultHandler.setDate(i, val.toString());
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-71: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type " + 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else if (values[i] instanceof Timestamp) {
                    Timestamp val = (Timestamp) values[i];
                    if (outputColumnTypes[i].equals("TIMESTAMP"))
                        resultHandler.setTimestamp(i, val.toString());
                    else
                        throw new ExaDataTypeException(
                            "E-UDF.CL.J-72: emit column '" + 
                            exaMetadata.getOutputColumnName(i) + 
                            "' is of type " + 
                            outputColumnTypes[i] + 
                            " but data given have type " + 
                            values[i].getClass().getCanonicalName()
                            );
                }
                else {
                    throw new ExaDataTypeException(
                        "E-UDF.CL.J-73: emit column '" + 
                        exaMetadata.getOutputColumnName(i) + 
                        "' is of unsupported type " + 
                        values[i].getClass().getCanonicalName()
                        );
                }

                String exMsg = resultHandler.checkException();
                if (exMsg != null && exMsg.length() > 0) {
                    throw new ExaIterationException("E-UDF.CL.J-74: "+exMsg);
                }
            }
        }else{
            if(exaMetadata.getOutputColumnCount()==1){
              resultHandler.setNull(0);
            }else{
              String errorText = "E-UDF.CL.J-75: emit() takes exactly " + exaMetadata.getOutputColumnCount();
              errorText += (exaMetadata.getOutputColumnCount() > 1) ? " arguments" : " argument";
              errorText += " (" + 1 + " given)";
              throw new ExaIterationException(errorText);
            }
        }

        boolean next = resultHandler.next();
        String exMsg = resultHandler.checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaIterationException("F-UDF.CL.J-76: "+exMsg);
        }
        if (!next) {
            throw new ExaIterationException("F-UDF.CL.J-77: Internal error while emiting row");
        }
    }

    @Override
    public Integer getInteger(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (!isConversionToIntegerSafe(object, columnNames.get(column)))
            return null;
        if (object instanceof Integer)
            return (Integer) object;
        else if (object instanceof Long)
            return new Integer(((Long) object).intValue());
        else if (object instanceof BigDecimal)
            return new Integer(((BigDecimal) object).intValueExact());
        else if (object instanceof Double)
            return new Integer(((Double) object).intValue());
        else
            throw 
              new ExaDataTypeException(
                "E-UDF.CL.J-78: getInteger cannot convert column '" + 
                columnNames.get(column) + 
                "' of type " + 
                exaMetadata.getInputColumnSqlType(column) + 
                " to an Integer"
                );
    }

    @Override
    public Integer getInteger(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-79: Column with name '" + name + "' does not exist");
        return getInteger(col);
    }

    @Override
    public Long getLong(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (!isConversionToLongSafe(object, columnNames.get(column)))
            return null;
        if (object instanceof Integer)
            return new Long(((Integer) object).longValue());
        else if (object instanceof Long)
            return (Long) object;
        else if (object instanceof BigDecimal)
            return new Long(((BigDecimal) object).longValueExact());
        else if (object instanceof Double)
            return new Long(((Double) object).longValue());
        else
            throw 
              new ExaDataTypeException(
                "E-UDF.CL.J-80: getLong cannot convert column '" + 
                columnNames.get(column) + 
                "' of type " + 
                exaMetadata.getInputColumnSqlType(column) + 
                " to a Long"
                );
    }

    @Override
    public Long getLong(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-81: Column with name '" + name + "' does not exist");
        return getLong(col);
    }

    @Override
    public BigDecimal getBigDecimal(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (object instanceof Integer)
            return new BigDecimal(((Integer) object).intValue());
        else if (object instanceof Long)
            return new BigDecimal(((Long) object).longValue());
        else if (object instanceof BigDecimal)
            return (BigDecimal) object;
        else if (object instanceof Double)
            return new BigDecimal(((Double) object).doubleValue());
        else
            throw 
              new ExaDataTypeException(
                "E-UDF.CL.J-82: getBigDecimal cannot convert column '" + 
                columnNames.get(column) + 
                "' of type " + 
                exaMetadata.getInputColumnSqlType(column) + 
                " to a BigDecimal"
                );
    }

    @Override
    public BigDecimal getBigDecimal(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-83: Column with name '" + name + "' does not exist");
        return getBigDecimal(col);
    }

    @Override
    public Double getDouble(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (object instanceof Integer)
            return new Double(((Integer)object).doubleValue());
        else if (object instanceof Long)
            return new Double(((Long)object).doubleValue());
        else if (object instanceof BigDecimal)
            return new Double(((BigDecimal)object).doubleValue());
        else if (object instanceof Double)
            return (Double) object;
        else
            throw 
              new ExaDataTypeException(
                  "E-UDF.CL.J-84: getDouble cannot convert column '" + 
                  columnNames.get(column) + 
                  "' of type " + 
                  exaMetadata.getInputColumnSqlType(column) + 
                  " to a Double"
                  );
    }

    @Override
    public Double getDouble(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-85: Column with name '" + name + "' does not exist");
        return getDouble(col);
    }

    @Override
    public String getString(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (object instanceof String)
            return (String) object;
        else if (object instanceof Integer || object instanceof Long || object instanceof BigDecimal || object instanceof Double ||
                    object instanceof Boolean || object instanceof Date || object instanceof Timestamp)
            return object.toString();
        else
            throw 
              new ExaDataTypeException(
                  "E-UDF.CL.J-86: getString cannot convert column '" + 
                  columnNames.get(column) + 
                  "' of type " + 
                  exaMetadata.getInputColumnSqlType(column) + 
                  " to a String"
                  );
    }

    @Override
    public String getString(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-87: Column with name '" + name + "' does not exist");
        return getString(col);
    }

    @Override
    public Boolean getBoolean(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (object instanceof Boolean)
            return (Boolean) object;
        else
            throw new ExaDataTypeException(
                "E-UDF.CL.J-88: getBoolean cannot convert column '" + 
                columnNames.get(column) + 
                "' of type " + 
                exaMetadata.getInputColumnSqlType(column) + 
                " to a Boolean"
                );
    }

    @Override
    public Boolean getBoolean(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-89: Column with name '" + name + "' does not exist");
        return getBoolean(col);
    }

    @Override
    public Date getDate(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (object instanceof Date)
            return (Date) object;
        else
            throw 
              new ExaDataTypeException(
                  "E-UDF.CL.J-90: getDate cannot convert column '" + 
                  columnNames.get(column) + 
                  "' of type " + 
                  exaMetadata.getInputColumnSqlType(column) + 
                  " to a Date");
    }

    @Override
    public Date getDate(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-91: Column with name '" + name + "' does not exist");
        return getDate(col);
    }

    @Override
    public Timestamp getTimestamp(int column) throws ExaIterationException, ExaDataTypeException {
        Object object = getObject(column);
        if (object == null)
            return null;
        if (object instanceof Timestamp)
            return (Timestamp) object;
        else
            throw 
              new ExaDataTypeException(
                  "E-UDF.CL.J-92: getTimestamp cannot convert column '" + 
                  columnNames.get(column) + 
                  "' of type " + 
                  exaMetadata.getInputColumnSqlType(column) + 
                  " to a Timestamp"
                  );
    }

    @Override
    public Timestamp getTimestamp(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-93: Column with name '" + name + "' does not exist");
        return getTimestamp(col);
    }

    @Override
    public Object getObject(int column) throws ExaIterationException, ExaDataTypeException {
        if (column < 0 || column >= exaMetadata.getInputColumnCount())
            throw new ExaIterationException("E-UDF.CL.J-94: Column number " + column + " does not exist");

        if (finished)
            throw new ExaIterationException("E-UDF.CL.J-95: Iteration finished");

        if (cache[column] != null)
            return cache[column];

        Object val = null;
        switch (inputColumnTypes[column]) {
            case "INT32":
                val = new Integer(tableIterator.getInt32(column));
                break;
            case "INT64":
                val = new Long(tableIterator.getInt64(column));
                break;
            case "NUMERIC":
                String numeric = tableIterator.getNumeric(column);
                if (!tableIterator.wasNull())
                    val = new BigDecimal(numeric);
                break;
            case "DOUBLE":
                val = new Double(tableIterator.getDouble(column));
                break;
            case "STRING":
                byte[] utf8Bytes = tableIterator.getString(column);
                try {
                    val = new String(utf8Bytes, "UTF-8");
                } catch (java.io.UnsupportedEncodingException ex) {
                    throw new ExaDataTypeException("F-UDF.CL.J-96: Column with name '" + columnNames.get(column) + "' contains invalid UTF-8 data");
                }
                break;
            case "BOOLEAN":
                val = new Boolean(tableIterator.getBoolean(column));
                break;
            case "DATE":
                String date = tableIterator.getDate(column);
                if (!tableIterator.wasNull())
                    val = Date.valueOf(date);
                break;
            case "TIMESTAMP":
                String timestamp = tableIterator.getTimestamp(column);
                if (!tableIterator.wasNull())
                    val = Timestamp.valueOf(timestamp);
                break;
            default:
                throw new ExaDataTypeException("F-UDF.CL.J-97: Column with name '" + columnNames.get(column) + "' has an invalid data type");
        }
        if (tableIterator.wasNull())
            val = null;
        cache[column] = val;
        return val;
    }

    @Override
    public Object getObject(String name) throws ExaIterationException, ExaDataTypeException {
        int col = columnNames.indexOf(name);
        if (col == -1)
            throw new ExaIterationException("E-UDF.CL.J-98: Column with name '" + name + "' does not exist");
        return getObject(col);
    }

    private boolean isConversionToIntegerSafe(Object from, String name) throws ExaDataTypeException {
        if (from == null)
            return false;

        if (from instanceof Long) {
            long val = (Long) from;
            if (val > Integer.MAX_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-99: emit column '" + name + "' has value of "
                            + val + " but column can only have maximum value of " + Integer.MAX_VALUE);
            if (val < Integer.MIN_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-100: emit column '" + name + "' has value of "
                            + val + " but column can only have minimum value of " + Integer.MIN_VALUE);
        }
        else if (from instanceof Float) {
            float val = (Float) from;
            if (val > Integer.MAX_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-101: emit column '" + name + "' has value of "
                            + val + " but column can only have maximum value of " + Integer.MAX_VALUE);
            if (val < Integer.MIN_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-102: emit column '" + name + "' has value of "
                            + val + " but column can only have minimum value of " + Integer.MIN_VALUE);
            if (val != Math.floor(val))
                throw new ExaDataTypeException("E-UDF.CL.J-103: emit column '" + name + "' has a non-integer value of " + val);
        }
        else if (from instanceof Double) {
            double val = (Double) from;
            if (val > Integer.MAX_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-112: emit column '" + name + "' has value of "
                            + val + " but column can only have maximum value of " + Integer.MAX_VALUE);
            if (val < Integer.MIN_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-104: emit column '" + name + "' has value of "
                            + val + " but column can only have minimum value of " + Integer.MIN_VALUE);
            if (val != Math.floor(val))
                throw new ExaDataTypeException("E-UDF.CL.J-105: emit column '" + name + "' has a non-integer value of " + val);
        }

        return true;
    }

    private boolean isConversionToLongSafe(Object from, String name) throws ExaDataTypeException {
        if (from == null)
            return false;

        if (from instanceof Float) {
            float val = (Float) from;
            if (val > Long.MAX_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-106: emit column '" + name + "' has value of "
                            + val + " but column can only have maximum value of " + Long.MAX_VALUE);
            if (val < Long.MIN_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-107: emit column '" + name + "' has value of "
                            + val + " but column can only have minimum value of " + Long.MIN_VALUE);
            if (val != Math.floor(val))
                throw new ExaDataTypeException("E-UDF.CL.J-108: emit column '" + name + "' has a non-integer value of " + val);
        }
        else if (from instanceof Double) {
            double val = (Double) from;
            if (val > Long.MAX_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-109: emit column '" + name + "' has value of "
                            + val + " but column can only have maximum value of " + Long.MAX_VALUE);
            if (val < Long.MIN_VALUE)
                throw new ExaDataTypeException("E-UDF.CL.J-110: emit column '" + name + "' has value of "
                            + val + " but column can only have minimum value of " + Long.MIN_VALUE);
            if (val != Math.floor(val))
                throw new ExaDataTypeException("E-UDF.CL.J-111: emit column '" + name + "' has a non-integer value of " + val);
        }

        return true;
    }

    private void clearCache() {
        for (int i = 0; i < cache.length; i++)
            cache[i] = null;
    }

    void setInsideRun(boolean inside) {
        insideRun = inside;
    }

    ExaIteratorImpl(ExaMetadata exaMetadata, TableIterator tableIterator, ResultHandler resultHandler) throws ExaIterationException {
        this.exaMetadata = exaMetadata;
        this.tableIterator = tableIterator;
        this.resultHandler = resultHandler;

        finished = false;
        singleInput = this.exaMetadata.getInputType().equals("SCALAR") ? true : false;
        singleOutput = this.exaMetadata.getOutputType().equals("RETURN") ? true : false;
        insideRun = false;
        cache = new Object[(int) this.exaMetadata.getInputColumnCount()];

        Metadata tmp = new Metadata();
        inputColumnTypes = new String[(int) this.exaMetadata.getInputColumnCount()];
        for (int i = 0; i < inputColumnTypes.length; i++) {
            inputColumnTypes[i] = tmp.inputColumnType(i).toString();
        }
        outputColumnTypes = new String[(int) this.exaMetadata.getOutputColumnCount()];
        for (int i = 0; i < outputColumnTypes.length; i++) {
            outputColumnTypes[i] = tmp.outputColumnType(i).toString();
        }

        columnNames = new ArrayList<String>();
        for(int i = 0; i < this.exaMetadata.getInputColumnCount(); i++) {
            columnNames.add(this.exaMetadata.getInputColumnName(i));
        }
    }
}
