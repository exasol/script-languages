#input_column: a,int64,INT,None,None,None
#input_column: b,string,VARCHAR(100),100,None,None
#input_type: SET

#output_column: a,int64,INT,None,None,None
#output_column: b,string,VARCHAR(100),100,None,None
#output_type: EMITS

def run(ctx):
        print "run"
        for i in range(1,10):
                print "a: "+str(ctx.a)
                print "b: "+ctx.b
                is_next=ctx.next()
                print "is_next: "+str(is_next)
