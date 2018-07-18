create external SCALAR SCRIPT
get_database_name() returns varchar(300) AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.database_name
/
create external scalar script
get_database_version() returns varchar(20) AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.database_version
/
create external scalar script
get_script_language() emits (s1 varchar(300), s2 varchar(300)) AS
# redirector @@redirector_url@@
def run(ctx):
	ctx.emit(exa.meta.script_language, "Python")
/
create external scalar script
get_script_name() returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.script_name
/
create external scalar script
get_script_schema() returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
        return exa.meta.script_schema
/
create external scalar script
get_current_user() returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
        return exa.meta.current_user
/
create external scalar script
get_current_schema() returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
        return exa.meta.current_schema
/
create external scalar script
get_script_code() returns varchar(2000) AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.script_code
/
create external scalar script
get_session_id() returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.session_id
/
create external scalar script
get_statement_id() returns number AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.statement_id
/
create external scalar script
get_node_count() returns number AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.node_count
/
create external scalar script
get_node_id() returns number AS
# redirector @@redirector_url@@
def run(ctx):
	return exa.meta.node_id
/
create external scalar script
get_vm_id() returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
 	return exa.meta.vm_id
/
create external scalar script
get_input_type_scalar() returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
 	return exa.meta.input_type
/
create external set script
get_input_type_set(a double) returns varchar(200) AS
# redirector @@redirector_url@@
def run(ctx):
 	return exa.meta.input_type
/
create external scalar script
get_input_column_count_scalar(c1 double, c2 varchar(100))
returns number as
# redirector @@redirector_url@@
def run(ctx):
 	return exa.meta.input_column_count
/
create external set script
get_input_column_count_set(c1 double, c2 varchar(100))
returns number as
# redirector @@redirector_url@@
def  run(ctx):
     return exa.meta.input_column_count
/
create external scalar script
get_input_columns(c1 double, c2 varchar(200))
emits (column_id number, column_name varchar(200), column_type varchar(20),
 	   column_sql_type varchar(20), column_precision number, column_scale number,
  	   column_length number) as
# redirector @@redirector_url@@
def run(ctx):
	cols = exa.meta.input_columns
 	for i in range(0, len(cols)):
		name  = cols[i].name
		precision = cols[i].precision
		thetype = repr(cols[i].type)
		sql_type = cols[i].sql_type
		scale = cols[i].scale
		length = cols[i].length
		if name == None: name = 'no-name'
		if thetype == None: thetype = 'no-type'
		if sql_type == None: sql_type = 'no-sql-type'
		if precision == None: precision = 0
		if scale == None: scale = 0
		if length == None: length = 0
		ctx.emit(i+1, name, thetype, sql_type, precision, scale, length)
/
create external scalar script
get_output_type_return()
returns varchar(200) as
# redirector @@redirector_url@@
def run(ctx):
 	return exa.meta.output_type
/
create external scalar script
get_output_type_emit()
emits (t varchar(200)) AS
# redirector @@redirector_url@@
def run(ctx):
   ctx.emit(exa.meta.output_type)
/
create external scalar script
get_output_column_count_return()
returns number AS
# redirector @@redirector_url@@
def run(ctx):
 	return exa.meta.output_column_count
/
create external scalar script
get_output_column_count_emit()
emits (x number, y number, z number) AS
# redirector @@redirector_url@@
def run(ctx):
	ctx.emit(exa.meta.output_column_count,exa.meta.output_column_count,exa.meta.output_column_count)
/
create external scalar script
get_output_columns()
emits (column_id number, column_name varchar(200), column_type varchar(20),
 	   column_sql_type varchar(20), column_precision number, column_scale number,
  	   column_length number) AS
# redirector @@redirector_url@@
def run(ctx):
	cols = exa.meta.output_columns
	for i in range(0, len(cols)):
		name = cols[i].name
		thetype = repr(cols[i].type)
		sql_type = cols[i].sql_type
		precision = cols[i].precision
		scale = cols[i].scale
		length = cols[i].length
		if name == None: name = 'no-name'
		if thetype == None: thetype = 'no-type'
		if sql_type == None: sql_type = 'no-sql-type'
		if precision == None: precision = 0
		if scale == None: scale = 0
		if length == None: length = 0
		ctx.emit(i+1, name, thetype, sql_type, precision, scale, length)
/

--  select get_output_columns() from dual;
-- (1, ...)

create external scalar script
get_precision_scale_length(n decimal(6,3), v varchar(10))
emits (precision1 number, scale1 number, length1 number, precision2 number, scale2 number, length2 number) AS
# redirector @@redirector_url@@
def run(ctx):
	v = exa.meta.input_columns[0]
  	precision1 = v.precision
  	scale1 = v.scale
	length1 = v.length
	w = exa.meta.input_columns[1]
	precision2 = w.precision
	scale2 = w.scale
	length2 = w.length
  	if precision1== None: precision1= 0 
  	if scale1== None: scale1= 0
	if length1== None: length1= 0
  	if precision2== None: precision2= 0 
  	if scale2== None: scale2= 0
	if length2== None: length2= 0
   	ctx.emit(precision1, scale1, length1, precision2, scale2, length2)
/

create external scalar script
get_char_length(text char(10))
emits(len1 number, len2 number, dummy char(20))
AS
# redirector @@redirector_url@@
def run(ctx):
	v = exa.meta.input_columns[0]
	w = exa.meta.output_columns[2]
	ctx.emit(v.length,w.length,'9876543210')
/

