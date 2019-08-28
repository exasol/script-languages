package com.exasol;

import java.util.List;
import java.util.Map;

/**
 * This class holds all information about a user-defined EXPORT, which can be
 * started using the {@code EXPORT INTO SCRIPT ...} statement.
 *
 * To support a user-defined EXPORT you can implement the callback method
 * generateSqlForExportSpec(ExaExportSpecification exportSpec). Please refer
 * to the user manual for more details.
 */
public interface ExaExportSpecification {

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
     * @return name of the connection, if one was specified.
     */
    public String getConnectionName();

    /**
     * This returns true if connection information was provided. The script can
     * then obtain the connection information via {@link #getConnectionInformation()}.
     *
     * @return true, if connection information was provided.
     */
    public boolean hasConnectionInformation();

    /**
     * Returns the connection information specified by the user.
     *
     * @return connection information.
     */
    public ExaConnectionInformation getConnectionInformation();

    /**
     * Returns the parameters specified in the EXPORT statement.
     *
     * @return parameters specified in the EXPORT statement.
     */
    public Map<String, String> getParameters();

    /**
     * This returns true if TRUNCATE was specified in the EXPORT statement.
     *
     * @return true, if TRUNCATE was specified.
     */
    public boolean hasTruncate();

    /**
     * This returns true if REPLACE was specified in the EXPORT statement.
     *
     * @return true, if REPLACE was specified.
     */
    public boolean hasReplace();

    /**
     * This returns true if CREATED BY was specified in the EXPORT statement.
     *
     * @return true, if CREATED BY was specified.
     */
    public boolean hasCreatedBy();

    /**
     * Returns the CREATED BY statement specified in the EXPORT statement
     * if {@link #hasCreatedBy()} is true.
     *
     * @return the CREATED BY statement, if one was specified.
     */
    public String getCreatedBy();

    /**
     * Returns the names of all columns in the EXPORT statement.
     *
     * @return LIST column names in the EXPORT statement.
     */
    public List<String> getSourceColumnNames();

}
