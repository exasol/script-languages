package com.exasol;


public class ExaConnectionInformationImpl implements ExaConnectionInformation {

    private ConnectionType type;
    private String address;
    private String user;
    private String password;

    public ExaConnectionInformationImpl(String type, String address, String user, String password)
    {
        if (type.equals("password")) {
            this.type = ConnectionType.PASSWORD;
        } else {
            throw new IllegalStateException("F-UDF.CL.SL.JAVA-1157: ExaConnectionInformationImpl: received unknown connection type: "+type);
        }
        this.address = address;
        this.user = user;
        this.password = password;
    }

    @Override
    public ConnectionType getType() {return type;}

    @Override
    public String getAddress() {return address;}

    @Override
    public String getUser() {return user;}

    @Override
    public String getPassword() {return password;}

}
