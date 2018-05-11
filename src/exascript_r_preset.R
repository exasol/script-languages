# Global R Configurations

options(digits.secs = 6)
options(error = function() traceback(2))
options(encoding = "UTF-8")

INTERNAL_META_OBJECT__ <- Metadata()
INTERNAL_INP_OBJECT__ <- NA
INTERNAL_OUT_OBJECT__ <- NA

exa <- {
    meta <- INTERNAL_META_OBJECT__
    m <- function(f, ...) {
        v <- f(meta, ...)
        msg <- Metadata_checkException(meta)
        if (!is.null(msg)) stop(msg) else v
    }
    mo <- list()
    mo[["database_name"]] = m(Metadata_databaseName)
    mo[["database_version"]] = m(Metadata_databaseVersion)
    mo[["script_name"]] = m(Metadata_scriptName)
    mo[["script_schema"]] = m(Metadata_scriptSchema)
    mo[["current_user"]] = m(Metadata_currentUser)
    mo[["current_schema"]] = m(Metadata_currentSchema)
    mo[["scope_user"]] = m(Metadata_scopeUser)
    mo[["script_code"]] = m(Metadata_scriptCode)
    mo[["session_id"]] = m(Metadata_sessionID_S)
    mo[["statement_id"]] = m(Metadata_statementID)
    mo[["node_count"]] = m(Metadata_nodeCount)
    mo[["node_id"]] = m(Metadata_nodeID)
    mo[["memory_limit"]] = m(Metadata_memoryLimit)
    mo[["vm_id"]] = m(Metadata_vmID_S)
    mo[["script_language"]] = paste(version$language, paste(version$major, version$minor, sep = "."))
    inpcols <- m(Metadata_inputColumnCount)
    getcoltype <- function(c, tbl) {
        if (tbl == "input") {
            colname <- m(Metadata_inputColumnName, c)
            coltype <- m(Metadata_inputColumnType, c)
            colprec <- m(Metadata_inputColumnPrecision, c)
            colscale <- m(Metadata_inputColumnScale, c)
            colsize <- m(Metadata_inputColumnSize, c)
            coltn <- m(Metadata_inputColumnTypeName, c)
        } else if (tbl == "output") {
            colname <- m(Metadata_outputColumnName, c)
            coltype <- m(Metadata_outputColumnType, c)
            colprec <- m(Metadata_outputColumnPrecision, c)
            colscale <- m(Metadata_outputColumnScale, c)
            colsize <- m(Metadata_outputColumnSize, c)
            coltn <- m(Metadata_outputColumnTypeName, c)
        }
        typestruct <- function(x, t, n, p, s, l)
            list(name = x, type = t, sql_type = n, precision = p, scale = s, length = l)
        if (coltype == "INT32") typestruct(colname, "integer", coltn, colprec, 0, NA)
        else if (coltype == "INT64") typestruct(colname, "double", coltn, colprec, 0, NA)
        else if (coltype == "DOUBLE") typestruct(colname, "double", coltn, NA, NA, NA)
        else if (coltype == "STRING") typestruct(colname, "character", coltn, NA, NA, colsize)
        else if (coltype == "BOOLEAN") typestruct(colname, "logical", coltn, NA, NA, NA)
        else if (coltype == "NUMERIC") typestruct(colname, "numeric", coltn, colprec, colscale, NA)
        else if (coltype == "DATE") typestruct(colname, "POSIXt", coltn, NA, NA, NA)
        else if (coltype == "TIMESTAMP") typestruct(colname, "POSIXt", coltn, NA, NA, NA)
        else typestruct(colname, coltype, coltn, colprec, colscale, colsize)
    }
    mo[["input_column_count"]] = inpcols
    mo[["input_columns"]] = lapply(0:(inpcols-1), function(c) getcoltype(c, "input"))
    mo[["input_type"]] = if (m(Metadata_inputType) == "EXACTLY_ONCE") "SCALAR" else "SET"
    outcols <- m(Metadata_outputColumnCount)
    mo[["output_column_count"]] = outcols
    mo[["output_columns"]] = lapply(0:(outcols-1), function(c) getcoltype(c, "output"))
    mo[["output_type"]] = if (m(Metadata_outputType) == "EXACTLY_ONCE") "RETURN" else "EMIT"
    cache <- list()
    import_script <- function(script) {
        v <- Metadata_moduleContent(meta, script)
        msg <- Metadata_checkException(meta)
        if (!is.null(msg)) stop(msg)
        else v
    }
    ret <- list(meta = mo)
    ret[["import_script"]] <- function (script) {
        code = import_script(script)
        if (is.null(cache[[code]])) {
            newmod <- new.env()
            assign("exa", ret, envir = newmod)
            cache[code] <<- list(newmod)
            eval(parse(text = code), newmod)
            newmod
        } else
            cache[[code]]
    }

    ret[["get_connection"]] <- function(connection_name) {
        wrapper <- Metadata_connectionInformation(meta, connection_name)
        msg <- Metadata_checkException(meta)
        if (!is.null(msg)) {
            stop(msg)
        }
        res <- list()
        res[["type"]] = ConnectionInformationWrapper_copyKind(wrapper)
        res[["address"]] = ConnectionInformationWrapper_copyAddress(wrapper)
        res[["user"]] = ConnectionInformationWrapper_copyUser(wrapper)
        res[["password"]] = ConnectionInformationWrapper_copyPassword(wrapper)
        return(res)
    }

    ret
}


INTERNAL_PARSE_WRAPPER__ <- function(code)
    eval(parse(text = code, encoding = "UTF-8"), globalenv())

# vim:ft=r:
