create or replace python scalar script print_connection(conn varchar (1000))
emits(type varchar(200), addr varchar(2000000), usr varchar(2000000), pwd varchar(2000000))
as
def run(ctx):
        c = exa.get_connection(ctx.conn)
        ctx.emit( c.type,  c.address,  c.user,  c.password )
/
