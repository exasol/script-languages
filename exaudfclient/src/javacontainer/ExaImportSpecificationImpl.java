package com.exasol;


import java.util.List;
import java.util.Map;


public class ExaImportSpecificationImpl implements ExaImportSpecification {


    private boolean isSubselect;
    private List<String> subselectColumnNames;
    private List<String> subselectColumnTypes;
    private String connectionName;
    private ExaConnectionInformation connectionInformation;
    private Map<String,String> parameters;
    
    public ExaImportSpecificationImpl(boolean isSubselect, List<String> subselectColumnNames, List<String> subselectColumnTypes, String connectionName, ExaConnectionInformation connectionInformation, Map<String,String> parameters) {
        this.isSubselect = isSubselect;
        this.subselectColumnNames = subselectColumnNames;
        this.subselectColumnTypes = subselectColumnTypes;
        this.connectionName = connectionName;
        this.connectionInformation = connectionInformation;
        this.parameters = parameters;
    }
    
    @Override
    public boolean isSubselect() {
        return isSubselect;
    }

    @Override
    public List<String> getSubselectColumnNames() {
        return subselectColumnNames;
    }

    // returns a string like "VARCHAR(100)"
    @Override
    public List<String> getSubselectColumnSqlTypes() {
        return subselectColumnTypes;
    }

    @Override
    public boolean hasConnectionName() {
        return connectionName != null;
    }

    @Override
    public String getConnectionName() {
        return connectionName;
    }

    @Override
    public boolean hasConnectionInformation() {
        return connectionInformation != null;
    }

    // same class as used for getConnection()
    @Override
    public ExaConnectionInformation getConnectionInformation() {
        return connectionInformation;
    }

    @Override
    public Map<String, String> getParameters() {
        return parameters;
    }
}
