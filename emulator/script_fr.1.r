#input_column: a,int64,INT,None,None,None
#input_column: b,string,VARCHAR(100),100,None,None
#input_type: SET

#output_column: a,int64,INT,None,None,None
#output_column: b,string,VARCHAR(100),100,None,None
#output_type: EMITS
run <- function(ctx) {
        print("run")
        # end_loop = FALSE
        # #while (!end_loop)
        # print(exa$meta)
        for(i in 1:100)
        {
                a=ctx$a
                b=ctx$b
                print(a)
                print(b)
                ctx$emit(a,b)
                exist_next=ctx$next_row(1)
                end_loop=!exist_next
                print(end_loop)
        }
}  
