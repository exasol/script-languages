#input_column: a,string,VARCHAR(100),100,None,None
#input_column: b,string,VARCHAR(100),100,None,None
#input_type: SCALAR

#output_column: b,string,VARCHAR(100),100,None,None
#output_type: EMITS
def run(ctx):
    conn = exa.get_connection('lsajklha')
    ctx.emit(str(conn))

