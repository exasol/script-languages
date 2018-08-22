CREATE LUA SCALAR SCRIPT
get_database_name() returns varchar(300) AS
function run(ctx)
	return exa.meta.database_name
end
/
create lua scalar script
get_database_version() returns varchar(20) as
function run(ctx)
	return exa.meta.database_version
end
/
create lua scalar script
get_script_language() emits (s1 varchar(300), s2 varchar(300)) as
function run(ctx)
	ctx.emit(exa.meta.script_language, "Lua")
end
/
create lua scalar script
get_script_name() returns varchar(200) as
function run(ctx)
	return exa.meta.script_name
end
/
create lua scalar script
get_script_schema() returns varchar(200) as
function run(ctx)
        return exa.meta.script_schema
end
/
create lua scalar script
get_current_user() returns varchar(200) as
function run(ctx)
        return exa.meta.current_user
end
/
create lua scalar script
get_scope_user() returns varchar(200) as
function run(ctx)
        return exa.meta.scope_user
end
/
create lua scalar script
get_current_schema() returns varchar(200) as
function run(ctx)
        return exa.meta.current_schema
end
/
create lua scalar script
get_script_code() returns varchar(2000) as
function run(ctx)
	return exa.meta.script_code
end
/
create lua scalar script
get_session_id() returns varchar(200) as
function run(ctx)
	return exa.meta.session_id
end
/
create lua scalar script
get_statement_id() returns number as
function run(ctx)
	return exa.meta.statement_id
end
/
create lua scalar script
get_node_count() returns number as
function run(ctx)
	return exa.meta.node_count
end
/
create lua scalar script
get_node_id() returns number as
function run(ctx)
	return exa.meta.node_id
end
/
create lua scalar script
get_vm_id() returns varchar(200) as
function run(ctx)
	return exa.meta.vm_id
end
/
create lua scalar script
get_input_type_scalar() returns varchar(200) as
function run(ctx)
	return exa.meta.input_type
end
/

-- Assert rows[0] = 'SCALAR'

create lua set script
get_input_type_set(a double) returns varchar(200) as
function run(ctx)
	return exa.meta.input_type
end
/

-- select get_input_type_set(x) from (values 1,2,3) as t(x);
-- Assert rows[0] = 'SET'



create lua scalar script
get_input_column_count_scalar(c1 double, c2 varchar(100))
returns number as
function run(ctx)
	return exa.meta.input_column_count
end
/

--  select get_input_column_count_scalar(12.3, 'hihihi') from dual;
-- Assert rows[0] = 2

create lua set script
get_input_column_count_set(c1 double, c2 varchar(100))
returns number as
function run(ctx)
    return exa.meta.input_column_count
end
/

-- select get_input_column_count_set(x, y) from (values (12.3, 'hihihi')) as t(x,y);
-- Assert rows[0] = 2 


create lua scalar script
get_input_columns(c1 double, c2 varchar(200))
emits (column_id number, column_name varchar(200), column_type varchar(20),
	   column_sql_type varchar(20), column_precision number, column_scale number,
 	   column_length number) as
function run(ctx)
	local id = 1
	for k,v in ipairs(exa.meta.input_columns) do
		local name = v.name
		local type = v.type
		local sql_type = v.sql_type
		local precision = v.precision
		local scale = v.scale
		local length = v.length
		
		if not name then name = 'no-name' end
		if not type then type = 'no-type' end
		if not sql_type then sql_type = 'no-sql-type' end
		if not precision then precision = 0 end
		if not scale then scale = 0 end
		if not length then length = 0 end

		ctx.emit(id, name, type, sql_type, precision, scale, length)	

		id = id + 1
	end
end
/

-- select get_input_columns(1.2, '123') from dual order by column_id;
-- Assert 

create lua scalar script
get_output_type_return()
returns varchar(200) as
function run(ctx)
	return exa.meta.output_type
end
/

-- select get_output_type_return() from dual;
-- Assert row[0] = 'RETURN'


create lua scalar script
get_output_type_emit()
emits (t varchar(200)) as
function run(ctx)
    ctx.emit(exa.meta.output_type)
end
/

-- select get_output_type_emit() from dual;
-- Assert row[0] = 'EMIT'



create lua scalar script
get_output_column_count_return()
returns number as
function run(ctx)
	return exa.meta.output_column_count
end
/
-- select get_output_column_count_return() from dual;
-- Assert row[0] = 1


create lua scalar script
get_output_column_count_emit()
emits (x number, y number, z number) as
function run(ctx)
	ctx.emit(exa.meta.output_column_count,exa.meta.output_column_count,exa.meta.output_column_count)
end
/

-- select get_output_column_count_emit() from dual;
-- Assert rows[0] = (3,3,3)

create lua scalar script
get_output_columns()
emits (column_id number, column_name varchar(200), column_type varchar(20),
	   column_sql_type varchar(20), column_precision number, column_scale number,
 	   column_length number) as
function run(ctx)
	local id = 1
	for k,v in ipairs(exa.meta.output_columns) do
		local name = v.name
		local type = v.type
		local sql_type = v.sql_type
		local precision = v.precision
		local scale = v.scale
		local length = v.length
		
		if not name then name = 'no-name' end
		if not type then type = 'no-type' end
		if not sql_type then sql_type = 'no-sql-type' end
		if not precision then precision = 0 end
		if not scale then scale = 0 end
		if not length then length = 0 end

		ctx.emit(id, name, type, sql_type, precision, scale, length)	

		id = id + 1
	end
end
/


create lua scalar script
get_char_length(text char(10))
emits(len1 number, len2 number, dummy char(20))
as
function run(ctx)
	local v = exa.meta.input_columns[1]
	local w = exa.meta.output_columns[3]
	ctx.emit(v.length,w.length,'9876543210')
end
/
-- LUA supports no decimal types
--  select get_output_columns() from dual;
-- (1, ...)

-- create lua scalar script
-- get_precision_scale(n decimal(6,3))
-- emits (precision number, scale number) as
-- function run(ctx)
-- 	local v = exa.meta.input_columns[1]
--     local precision = v.precision
--     local scale = v.scale
-- 	if not precision then precision = 0 end
-- 	if not scale then scale = 0 end
--     ctx.emit(precision, scale)
-- end
-- /
