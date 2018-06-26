#input_column: a,int64,INT,None,None,None
#input_column: b,string,VARCHAR(100),100,None,None
#input_type: SET

#output_column: a,int64,INT,None,None,None
#output_column: b,string,VARCHAR(100),100,None,None
#output_type: EMITS
#import tensorflow as tf

def run(ctx):
    while True:
        ctx.emit(ctx.a * 100, "@" + repr(ctx.b))
        if not ctx.next(): break
