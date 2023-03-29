
CREATE LUA SET SCRIPT VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
  function run(ctx)
    ctx.emit(1)
  end
/

CREATE LUA SCALAR SCRIPT VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
  function run(ctx)
    ctx.emit(1)
  end
/

CREATE LUA SCALAR SCRIPT VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
  function run(ctx)
    ctx.emit(1)
  end
/

CREATE LUA SET SCRIPT VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
  function run(ctx)
    ctx.emit(1)
  end
/

CREATE LUA SET SCRIPT VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
  function run(ctx)
    local outRec = {}
    for col = 1, exa.meta.output_column_count do
      outRec[col] = ctx[1]
    end
    ctx.emit(unpack(outRec))
  end
/

CREATE LUA SET SCRIPT VAREMIT_ALL_GENERIC (...) EMITS (...) AS
function run(ctx)
  local outRec = {}
  for col = 1, exa.meta.output_column_count do
    outRec[col] = ctx[1]
  end
  ctx.emit(unpack(outRec))
end
/

CREATE LUA SET SCRIPT VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
function run(data)
    data.emit(exa.meta.output_column_count, 1)
    for i=1,exa.meta.output_column_count do
        data.emit(exa.meta.output_columns[i].name, 1)
        data.emit(exa.meta.output_columns[i].type, 1)
        data.emit(exa.meta.output_columns[i].sql_type, 1)
        data.emit(exa.meta.output_columns[i].precision, 1)
        data.emit(exa.meta.output_columns[i].scale, 1)
        data.emit(exa.meta.output_columns[i].length, 1)
    end
end
/

CREATE LUA SET SCRIPT VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
function run(ctx)
  ctx.emit(1)
end
/

CREATE LUA SET SCRIPT VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
function run(ctx)
  return 1
end
/

CREATE LUA SET SCRIPT VAREMIT_EMIT_INPUT (...) EMITS (...) AS
function run(ctx)
  local outRec = {}
  for col = 1, exa.meta.output_column_count do
    outRec[col] = ctx[col]
  end
  ctx.emit(unpack(outRec))
end
/

CREATE LUA SET SCRIPT VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
function run(ctx)
  local outRec = {}
  for col = 1, exa.meta.output_column_count do
    assert (exa.meta.input_columns[col].sql_type == exa.meta.output_columns[col].sql_type)
    outRec[col] = ctx[col]
  end
  ctx.emit(unpack(outRec))
end
/




-- ---------------------------------------------------
-- Now the same as above but with default_output_columns  
-- ---------------------------------------------------

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SET (a double) EMITS (...) AS
  function run(ctx)
    ctx.emit(1)
  end
  function default_output_columns()
      return "x double"
  end
/

CREATE LUA SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_SCALAR (a double) EMITS (...) AS
  function run(ctx)
    ctx.emit(1)
  end
  function default_output_columns()
    return "x double"
  end
/

CREATE LUA SCALAR SCRIPT DEFAULT_VAREMIT_SIMPLE_ALL_DYN (...) EMITS (...) AS
  function run(ctx)
    ctx.emit(1)
  end
  function default_output_columns()
    return "x double"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_SYNTAX_VAR (...) EMITS ( ...   ) AS
  function run(ctx)
    ctx.emit(1)
  end
  function default_output_columns()
    return "x double"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_GENERIC_EMIT (a varchar(100)) EMITS (...) AS
  function run(ctx)
    local outRec = {}
    for col = 1, exa.meta.output_column_count do
      outRec[col] = ctx[1]
    end
    ctx.emit(unpack(outRec))
  end
  function default_output_columns()
    return "a varchar(100)"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_ALL_GENERIC (...) EMITS (...) AS
function run(ctx)
  local outRec = {}
  for col = 1, exa.meta.output_column_count do
    outRec[col] = ctx[1]
  end
  ctx.emit(unpack(outRec))
end
  function default_output_columns()
    return "a varchar(100)"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_METADATA_SET_EMIT (...) EMITS(...) AS
function run(data)
    data.emit(exa.meta.output_column_count, 1)
    for i=1,exa.meta.output_column_count do
        data.emit(exa.meta.output_columns[i].name, 1)
        data.emit(exa.meta.output_columns[i].type, 1)
        data.emit(exa.meta.output_columns[i].sql_type, 1)
        data.emit(exa.meta.output_columns[i].precision, 1)
        data.emit(exa.meta.output_columns[i].scale, 1)
        data.emit(exa.meta.output_columns[i].length, 1)
    end
end
  function default_output_columns()
    return "a varchar(123), b double"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_NON_VAR_EMIT (...) EMITS (a double) AS
function run(ctx)
  ctx.emit(1)
end
  function default_output_columns()
    return "x double"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_SIMPLE_RETURNS (a int) RETURNS int AS
function run(ctx)
  return 1
end
  function default_output_columns()
    return "x double"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT (...) EMITS (...) AS
function run(ctx)
  local outRec = {}
  for col = 1, exa.meta.output_column_count do
    outRec[col] = ctx[col]
  end
  ctx.emit(unpack(outRec))
end
  function default_output_columns()
    return "x varchar(100), y varchar(100), z varchar(100)"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_EMIT_INPUT_WITH_META_CHECK (...) EMITS (...) AS
function run(ctx)
  local outRec = {}
  for col = 1, exa.meta.output_column_count do
    assert (exa.meta.input_columns[col].sql_type == exa.meta.output_columns[col].sql_type)
    outRec[col] = ctx[col]
  end
  ctx.emit(unpack(outRec))
end
  function default_output_columns()
    return "x varchar(100), y varchar(100), z varchar(100)"
  end
/

CREATE LUA SET SCRIPT DEFAULT_VAREMIT_EMPTY_DEF(X DOUBLE) EMITS (...) AS
function run(ctx)
    ctx.emit(1.4)
end

function default_output_columns()
   return ''
end
/


CREATE LUA SET SCRIPT OUTPUT_COLUMNS_AS_IN_CONNECTION_SPOT4245 (a double) EMITS (...) AS
  function run(ctx)
    ctx.emit(1.0,1.0,1.0)
  end
  function default_output_columns()
      local c = exa.get_connection('SPOT4245')
      return c.type .. " double, " .. c.address .. " double, " .. c.user .. " double, " .. c.password .. " double"
  end
/
