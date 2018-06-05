#input_column: a,string,VARCHAR(100),100,None,None
#input_column: b,string,VARCHAR(100),100,None,None
#input_type: SET

#output_column: b,string,VARCHAR(100),100,None,None
#output_type: EMITS
#!/usr/bin/swipl -q
:- use_module(library(csv)).
:- initialization(main,main).


process(row(A,B),row(A)).

f(O) :- csv_read_row(user_input,R,O), R \= end_of_file, !, process(R,RR), csv_write_stream(user_output, [RR], []), f(O).
f(_).



main(_Argv) :- prompt(_, ''), csv_options(O,[]), f(O).
