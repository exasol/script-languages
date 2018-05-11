package com.exasol;


import java.util.List;
import java.util.Map;


public class ExaExportSpecificationImpl implements ExaExportSpecification {


    private String connectionName;
    private ExaConnectionInformation connectionInformation;
    private Map<String,String> parameters;
    private boolean hasTruncate;
    private boolean hasReplace;
    private String createdBy;
    private List<String> sourceColumnNames;
    
    public ExaExportSpecificationImpl(String connectionName, ExaConnectionInformation connectionInformation, Map<String,String> parameters,
                                      boolean hasTruncate, boolean hasReplace, String createdBy,
                                      List<String> sourceColumnNames) {
        this.connectionName = connectionName;
        this.connectionInformation = connectionInformation;
        this.parameters = parameters;
        this.hasTruncate = hasTruncate;
        this.hasReplace = hasReplace;
        this.createdBy = createdBy;
        this.sourceColumnNames = sourceColumnNames;
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

    @Override
    public boolean hasTruncate() {
        return hasTruncate;
    }

    @Override
    public boolean hasReplace() {
        return hasReplace;
    }

    @Override
    public boolean hasCreatedBy() {
        return createdBy != null;
    }

    @Override
    public String getCreatedBy() {
        return createdBy;
    }

    @Override
    public List<String> getSourceColumnNames() {
        return sourceColumnNames;
    }

}
