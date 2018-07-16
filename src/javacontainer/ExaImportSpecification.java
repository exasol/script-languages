package com.exasol;

import java.util.List;
import java.util.Map;

/**
 * This class holds all information about an user defined IMPORT, which can be
 * started using the {@code IMPORT FROM SCRIPT ...} statement.
 *
 * To support a user defined IMPORT you can implement the callback method
 * generateSqlForImportSpec(ExaImportSpecification importSpec). Please refer
 * to the user manual for more details.
 */
public interface ExaImportSpecification {

    /**
     * Indicates whether this IMPORT is a subselect.
     *
     * @return true, if the IMPORT is used inside a SELECT statement
     *         (i.e. not inside an IMPORT INTO table statement), false, otherwise
     */
    public boolean isSubselect();

    /**
     * Returns the names of all specified columns if the user specified the
     * target column names and types (only relevant if isSubselect() is true)
     *
     * @return List of names of the specified target columns
     */
    public List<String> getSubselectColumnNames();

    /**
     * Returns the types of all specified columns if the user specified the
     * target column names and types (only relevant if isSubselect() is true).
     * The types are returned in SQL format (e.g. "VARCHAR(100)").
     *
     * @return List of names of the specified target columns
     */
    public List<String> getSubselectColumnSqlTypes();

    /**
     * Indicates whether the name of a connection was specified. The script can
     * then obtain the connection information via ExaMetadata.getConnection(name).
     *
     * @return true, if the name of a connection was specified.
     */
    public boolean hasConnectionName();

    /**
     * Returns the name of the specified connection if {@link #hasConnectionName()} is true.
     *
     * @return name of the connection, if one was specified
     */
    public String getConnectionName();

    /**
     * This returns true if connection information were provided. The script can
     * then obtain the connection information via {@link #getConnectionInformation()}.
     *
     * @return true, if connection information were provided.
     */
    public boolean hasConnectionInformation();

    /**
     * Returns the connection information specified by the user.
     *
     * @return connection information
     */
    public ExaConnectionInformation getConnectionInformation();

    /**
     * Returns the parameters specified in the IMPORT statement.
     *
     * @return parameters specified in the IMPORT statement
     */
    public Map<String, String> getParameters();
}
