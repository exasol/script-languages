#input_column: b,string,VARCHAR(100),100,None,None
#input_type: SET

#output_column: b,string,VARCHAR(100),100,None,None
#output_type: EMITS
#!/bin/bash
ls -l /tmp
echo "This is an error message" 1>&2
ls -l /tmp
