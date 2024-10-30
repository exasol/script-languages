CREATE or replace java SCALAR SCRIPT print_connection(conn varchar (1000))
emits(type varchar(200), addr varchar(2000000), usr varchar(2000000), pwd varchar(2000000))
as
%jvmoption -Xms64m -Xmx128m -Xss512k;
class PRINT_CONNECTION {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ExaConnectionInformation c = exa.getConnection(ctx.getString("conn"));
        ctx.emit(c.getType().toString().toLowerCase(),c.getAddress(),c.getUser(), c.getPassword());
    }
}
/

CREATE or replace java SCALAR SCRIPT print_connection_v2(conn varchar (1000))
emits(type varchar(200), addr varchar(2000000), usr varchar(2000000), pwd varchar(2000000))
as
%env SCRIPT_OPTIONS_PARSER_VERSION=2;
%jvmoption -Xms64m -Xmx128m -Xss512k;
class PRINT_CONNECTION_V2 {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ExaConnectionInformation c = exa.getConnection(ctx.getString("conn"));
        ctx.emit(c.getType().toString().toLowerCase(),c.getAddress(),c.getUser(), c.getPassword());
    }
}
/

CREATE or replace java SET SCRIPT print_connection_set_emits(conn varchar (1000))
emits(type varchar(200), addr varchar(2000000), usr varchar(2000000), pwd varchar(2000000))
as
%jvmoption -Xms64m -Xmx128m -Xss512k;
class PRINT_CONNECTION_SET_EMITS {
    static void run(ExaMetadata exa, ExaIterator ctx) throws Exception {
        ExaConnectionInformation c = exa.getConnection(ctx.getString("conn"));
        ctx.emit(c.getType().toString().toLowerCase(),c.getAddress(),c.getUser(), c.getPassword());
    }
}
/
