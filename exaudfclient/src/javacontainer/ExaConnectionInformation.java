package com.exasol;

/**
 * This class holds all information about a specific connection, as it can be created using
 * the CREATE CONNECTION statement.
 */
public interface ExaConnectionInformation {

    public enum ConnectionType {
        PASSWORD
    }

    /**
     * @return type of the connection.
     */
    public ConnectionType getType();

    /**
     * @return address of the connection, i.e. the part that
     *         follows the TO keyword in the CREATE CONNECTION command.
     */
    public String getAddress();

    /**
     * @return username of the connection, i.e. the part that
     *         follows the USER keyword in the CREATE CONNECTION command.
     */
    public String getUser();

    /**
     * @return password of the connection, i.e. the part that
     *         follows the IDENTIFIED BY keyword in the CREATE CONNECTION command.
     */
    public String getPassword();

}
