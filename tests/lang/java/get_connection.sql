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
