create or replace python set script impal_use_is_subselect(...) emits (x varchar(2000)) as
def generate_sql_for_import_spec(import_spec):
        return "select " + str(import_spec.is_subselect)
/

create or replace python scalar script impal_use_param_foo_bar(...) returns varchar(2000) as
def generate_sql_for_import_spec(import_spec):
        return "select '" +  import_spec.parameters['FOO']  + "', '" + import_spec.parameters['BAR'] + "'"
/

create or replace python set script impal_use_connection_name(...) emits (x varchar(2000)) as
def generate_sql_for_import_spec(import_spec):
    return "select '" + import_spec.connection_name + "'"
/


create or replace python set script impal_use_connection(...) emits (x varchar(2000)) as
def generate_sql_for_import_spec(import_spec):
        return "select '" + import_spec.connection.user + import_spec.connection.password + import_spec.connection.address + import_spec.connection.type + "'"
/

-- create or replace python set script impal_use_connection_fooconn(...) emits (x varchar(2000)) as
-- def generate_sql_for_import_spec(import_spec):
-- 	c = exa.get_connection('FOOCONN')
--         return "select '" + str(c.address) + str(c.user) + str(c.password) + "'"
-- /


create or replace python set script impal_use_all(...) emits (x varchar(2000)) as
def generate_sql_for_import_spec(import_spec):
        is_sub = str(import_spec.is_subselect).upper()
        connection_string = 'X'
        connection_name = 'Y'
        foo = 'Z'
        types = 'T'
        names = 'N'
        if import_spec.connection is not None:
                connection_string = import_spec.connection.user + import_spec.connection.password + import_spec.connection.address + import_spec.connection.type
        if import_spec.connection_name is not None:
                connection_name = import_spec.connection_name
        if import_spec.parameters['FOO'] is not None:
                foo = import_spec.parameters['FOO']
        if import_spec.subselect_column_types is not None:
                for i in range(0, len(import_spec.subselect_column_types)):
                        types = types + import_spec.subselect_column_types[i]
                        names = names + import_spec.subselect_column_names[i]
        return "select 1, '" + is_sub + '_' + connection_name + '_' + connection_string + '_' +  foo + '_' + types + '_' + names + "'"
/
