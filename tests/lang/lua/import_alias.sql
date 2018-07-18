create or replace lua set script impal_use_is_subselect(...) emits (x varchar(2000)) as
  function generate_sql_for_import_spec(import_spec)
    return "select " .. import_spec.is_subselect
  end
/

create or replace lua scalar script impal_use_param_foo_bar(...) returns varchar(2000) as
  function generate_sql_for_import_spec(import_spec)
    return "select '" ..  import_spec.parameters.FOO  .. "', '" .. import_spec.parameters.BAR .. "'"
  end
/

create or replace lua set script impal_use_connection_name(...) emits (x varchar(2000)) as
  function generate_sql_for_import_spec(import_spec)
    return "select '" .. import_spec.connection_name .. "'"
  end
/

create or replace lua set script impal_use_connection(...) emits (x varchar(2000)) as
  function generate_sql_for_import_spec(import_spec)
    return "select '" .. import_spec.connection.user .. import_spec.connection.password .. import_spec.connection.address .. import_spec.connection.type .. "'"
  end
/

create or replace lua set script impal_use_all(...) emits (x varchar(2000)) as
  function generate_sql_for_import_spec(import_spec)
    local is_sub = string.upper(import_spec.is_subselect)
    local connection_string = 'X'
    local connection_name = 'Y'
    local foo = 'Z'
        local types = 'T'
        local names = 'N'
    if import_spec.connection ~= nil then
        connection_string = import_spec.connection.user .. import_spec.connection.password .. import_spec.connection.address .. import_spec.connection.type
        end
    if import_spec.connection_name ~= nil then
        connection_name = import_spec.connection_name
    end
    if import_spec.parameters.FOO ~= nil then
                foo = import_spec.parameters.FOO
    end
        if import_spec.subselect_column_types ~= nil then
                for i=1, #import_spec.subselect_column_types do
                types = types .. import_spec.subselect_column_types[i]
                        names = names .. import_spec.subselect_column_names[i]
                end
        end
    return "select 1, '" .. is_sub .. '_' .. connection_name .. '_' .. connection_string .. '_' ..  foo .. '_' .. types .. '_' .. names .. "'"
  end
/
