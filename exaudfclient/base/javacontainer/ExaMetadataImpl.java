package com.exasol;

import java.math.BigInteger;
import java.util.ArrayList;
import com.exasol.swig.Metadata;
import com.exasol.swig.ConnectionInformationWrapper;

class ExaMetadataImpl implements ExaMetadata {
    private Metadata metadata;

    private String databaseName;
    private String databaseVersion;
    private String scriptLanguage;
    private String scriptName;
    private String currentUser;
    private String scopeUser;
    private String currentSchema;
    private String scriptSchema;
    private String scriptCode;
    private String sessionId;
    private long statementId;
    private long nodeCount;
    private long nodeId;
    private String vmId;
    private BigInteger memoryLimit;
    private String inputType;
    private long inputColumnCount;
    private ColumnInfo[] inputColumns;
    private String outputType;
    private long outputColumnCount;
    private ColumnInfo[] outputColumns;
    private ArrayList<String> importedScripts;

    @Override
    public String getDatabaseName() { return databaseName; }
    @Override
    public String getDatabaseVersion() { return databaseVersion; }
    @Override
    public String getScriptLanguage() { return scriptLanguage; }
    @Override
    public String getScriptName() { return scriptName; }
    @Override
    public String getCurrentUser() { return currentUser; }
    @Override
    public String getScopeUser() { return scopeUser; }
    @Override
    public String getCurrentSchema() { return currentSchema; }
    @Override
    public String getScriptSchema() { return scriptSchema; }
    @Override
    public String getScriptCode() { return scriptCode; }
    @Override
    public String getSessionId() { return sessionId; }
    @Override
    public long getStatementId() { return statementId; }
    @Override
    public long getNodeCount() { return nodeCount; }
    @Override
    public long getNodeId() { return nodeId; }
    @Override
    public String getVmId() { return vmId; }
    @Override
    public BigInteger getMemoryLimit() { return memoryLimit; }
    @Override
    public String getInputType() { return inputType; }
    @Override
    public long getInputColumnCount() {return inputColumnCount; }
    @Override
    public String getInputColumnName(int column) throws ExaIterationException {
        if (column < 0 || column >= inputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-114: Column number " + column + " does not exist");
        return inputColumns[column].name;
    }
    @Override
    public Class<?> getInputColumnType(int column) throws ExaIterationException {
        if (column < 0 || column >= inputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-115: Column number " + column + " does not exist");
        return inputColumns[column].type;
    }
    @Override
    public String getInputColumnSqlType(int column) throws ExaIterationException {
        if (column < 0 || column >= inputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-116: Column number " + column + " does not exist");
        return inputColumns[column].sqlType;
    }
    @Override
    public long getInputColumnPrecision(int column) throws ExaIterationException {
        if (column < 0 || column >= inputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-117: Column number " + column + " does not exist");
        return inputColumns[column].precision;
    }
    @Override
    public long getInputColumnScale(int column) throws ExaIterationException {
        if (column < 0 || column >= inputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-118: Column number " + column + " does not exist");
        return inputColumns[column].scale;
    }
    @Override
    public long getInputColumnLength(int column) throws ExaIterationException {
        if (column < 0 || column >= inputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-119: Column number " + column + " does not exist");
        return inputColumns[column].length;
    }
    @Override
    public String getOutputType() { return outputType; }
    @Override
    public long getOutputColumnCount() { return outputColumnCount; }
    @Override
    public String getOutputColumnName(int column) throws ExaIterationException {
        if (column < 0 || column >= outputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-120: Column number " + column + " does not exist");
        return outputColumns[column].name;
    }
    @Override
    public Class<?> getOutputColumnType(int column) throws ExaIterationException {
        if (column < 0 || column >= outputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-121: Column number " + column + " does not exist");
        return outputColumns[column].type;
    }
    @Override
    public String getOutputColumnSqlType(int column) throws ExaIterationException {
        if (column < 0 || column >= outputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-122: Column number " + column + " does not exist");
        return outputColumns[column].sqlType;
    }
    @Override
    public long getOutputColumnPrecision(int column) throws ExaIterationException {
        if (column < 0 || column >= outputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-123: Column number " + column + " does not exist");
        return outputColumns[column].precision;
    }
    @Override
    public long getOutputColumnScale(int column) throws ExaIterationException {
        if (column < 0 || column >= outputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-124: Column number " + column + " does not exist");
        return outputColumns[column].scale;
    }
    @Override
    public long getOutputColumnLength(int column) throws ExaIterationException {
        if (column < 0 || column >= outputColumns.length)
            throw new ExaIterationException("E-UDF.CL.J-125: Column number " + column + " does not exist");
        return outputColumns[column].length;
    }
    
    @Override
    public Class<?> importScript(String name) throws ExaCompilationException, ClassNotFoundException {
        if (name == null)
            throw new ExaCompilationException("F-UDF.CL.J-126: Script name is null");
        boolean isQuoted = (name.charAt(0) == '"' && name.charAt(name.length() - 1) == '"');
        String scriptName = isQuoted ? name.substring(1, name.length() - 1) : name.toUpperCase();
        if (!importedScripts.contains(scriptName)) {
            String code = metadata.moduleContent(name);
            code = "package com.exasol;\r\n" + code;
            String exMsg = checkException();
            if (exMsg != null && exMsg.length() > 0) {
                throw new ExaCompilationException("F-UDF.CL.J-127: "+exMsg);
            }
            try{
                ExaCompiler.compile("com.exasol." + scriptName, code);
            }catch(ExaCompilationException ex){
                throw new ExaCompilationException("F-UDF.CL.J-128: "+ex.toString());
            }
            importedScripts.add(scriptName);
        }
        return Class.forName("com.exasol." + scriptName);
    }
    
    @Override
    public ExaConnectionInformation getConnection(String name) throws ExaConnectionAccessException {
        if (name == null) {
            throw new ExaConnectionAccessException("E-UDF.CL.J-129: Connection name is null");
        }
        boolean isQuoted = (name.charAt(0) == '"' && name.charAt(name.length() - 1) == '"');
        String connectionName = isQuoted ? name.substring(1, name.length() - 1) : name.toUpperCase();
        ConnectionInformationWrapper w = metadata.connectionInformation(connectionName);
        String exMsg = checkException();
        if (exMsg != null && exMsg.length() > 0) {
            throw new ExaConnectionAccessException("E-UDF.CL.J-130: "+exMsg);
        }
        return new ExaConnectionInformationImpl(w.copyKind(), w.copyAddress(), w.copyUser(), w.copyPassword());
    }

    public String checkException() {
        return metadata.checkException();
    }

    ExaMetadataImpl() throws ClassNotFoundException, ExaDataTypeException {
        metadata = new Metadata();

        databaseName = metadata.databaseName();
        databaseVersion = metadata.databaseVersion();
        scriptLanguage = "Java " + System.getProperty("java.version");
        scriptName = metadata.scriptName();
        scriptSchema = metadata.scriptSchema();
        currentUser = metadata.currentUser();
        scopeUser = metadata.scopeUser();
        currentSchema = metadata.currentSchema();
        scriptCode = metadata.scriptCode();
        sessionId = metadata.sessionID_S();
        statementId = metadata.statementID();
        nodeCount = metadata.nodeCount();
        nodeId = metadata.nodeID();
        vmId = metadata.vmID_S();
        memoryLimit = metadata.memoryLimit();

        inputType = (metadata.inputType().toString().equals("EXACTLY_ONCE")) ? "SCALAR" : "SET";
        inputColumnCount = metadata.inputColumnCount();
        inputColumns = new ColumnInfo[(int) inputColumnCount];
        for (int i = 0; i < inputColumns.length; i++) {
            inputColumns[i] = new ColumnInfo();
            inputColumns[i].initInputColumnInfo(i);
        }

        outputType = (metadata.outputType().toString().equals("EXACTLY_ONCE")) ? "RETURN" : "EMIT";
        outputColumnCount = metadata.outputColumnCount();
        outputColumns = new ColumnInfo[(int) outputColumnCount];
        for (int i = 0; i < outputColumns.length; i++) {
            outputColumns[i] = new ColumnInfo();
            outputColumns[i].initOutputColumnInfo(i);
        }

        importedScripts = new ArrayList<String>();
    }

    private class ColumnInfo {
        private String name;
        private String exaType;
        private Class type;
        private String sqlType;
        private long precision;
        private long scale;
        private long length;

        private void initInputColumnInfo(long column) throws ClassNotFoundException, ExaDataTypeException {
            name = metadata.inputColumnName(column);
            exaType = metadata.inputColumnType(column).toString();
            sqlType = metadata.inputColumnTypeName(column);
            precision = metadata.inputColumnPrecision(column);
            scale = metadata.inputColumnScale(column);
            length = metadata.inputColumnSize(column);
            setColumnInfo();
        }

        private void initOutputColumnInfo(long column) throws ClassNotFoundException, ExaDataTypeException {
            name = metadata.outputColumnName(column);
            exaType = metadata.outputColumnType(column).toString();
            sqlType = metadata.outputColumnTypeName(column);
            precision = metadata.outputColumnPrecision(column);
            scale = metadata.outputColumnScale(column);
            length = metadata.outputColumnSize(column);
            setColumnInfo();

        }

        private void setColumnInfo() throws ClassNotFoundException, ExaDataTypeException {
            switch(exaType) {
                case "INT32":
                    type = Class.forName("java.lang.Integer");
                    scale = 0;
                    length = 0;
                    break;
                case "INT64":
                    type = Class.forName("java.lang.Long");
                    scale = 0;
                    length = 0;
                    break;
                case "NUMERIC":
                    type = Class.forName("java.math.BigDecimal");
                    length = 0;
                    break;
                case "DOUBLE":
                    type = Class.forName("java.lang.Double");
                    precision = 0;
                    scale = 0;
                    length = 0;
                    break;
                case "STRING":
                    type = Class.forName("java.lang.String");
                    precision = 0;
                    scale = 0;
                    break;
                case "BOOLEAN":
                    type = Class.forName("java.lang.Boolean");
                    precision = 0;
                    scale = 0;
                    length = 0;
                    break;
                case "DATE":
                    type = Class.forName("java.sql.Date");
                    precision = 0;
                    scale = 0;
                    length = 0;
                    break;
                case "TIMESTAMP":
                    type = Class.forName("java.sql.Timestamp");
                    precision = 0;
                    scale = 0;
                    length = 0;
                    break;
                default:
                    throw new ExaDataTypeException("F-UDF.CL.J-131: data type " + exaType + " is not supported");
            }
        }
    }
}
