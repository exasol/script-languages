import sys
unicode = str
decodeUTF8 = lambda x: x
encodeUTF8 = lambda x: x

from exascript_python import *
import decimal
import datetime
import imp

class ExaUDFError(Exception):
    pass 

def clean_stacktrace_line(line):
    import re
    match = re.match("""^\s+File "(.+)", line ([0-9]+), in (.+)$""",line)
    if match is not None:
        filename=match.group(1)
        lineno=match.group(2)
        text=match.group(3)
        if filename!="<EXASCRIPTPP>" and filename!="<EXASCRIPT>":
            return "{filename}:{lineno} {text}\n".format(
                            filename=filename,
                            lineno=lineno,
                            text=text)
        else:
            return ""
    elif "Traceback (most recent call last):" in line:
        return ""
    else:
        return line

def create_clean_stacktrace(etype, evalue, etb):
    import traceback as tb
    st=tb.format_exception(etype, evalue, etb)
    new_st=[clean_stacktrace_line(line) for line in st]
    return "".join(new_st)

def create_exception_with_complete_backtrace(error_code, error_message, exc_info):
    exception_type = exc_info[0]
    exception_message = exc_info[1]
    exception_bt = exc_info[2]
    backtrace = create_clean_stacktrace(exception_type,exception_message,exception_bt)
    new_exception_message = "%s: %s \n%s" % (error_code, error_message, backtrace)
    print(new_exception_message)
    return ExaUDFError(new_exception_message)

class exa:
    def __init__(self):
        self.__modules = {}
        self.__meta = meta = Metadata()
        class exameta: pass
        mo = exameta()
        mo.database_name = meta.databaseName()
        mo.database_version = meta.databaseVersion()
        mo.script_name = decodeUTF8(meta.scriptName())
        mo.script_schema = decodeUTF8(meta.scriptSchema())
        mo.current_user = decodeUTF8(meta.currentUser())
        mo.scope_user = decodeUTF8(meta.scopeUser())
        mo.current_schema = decodeUTF8(meta.currentSchema())
        mo.scope_user = decodeUTF8(meta.scopeUser())
        mo.script_code = decodeUTF8(meta.scriptCode())
        if type(sys.version_info) == tuple:
            mo.script_language = "Python %d.%d.%d" % (sys.version_info[0], sys.version_info[1], sys.version_info[2])
        else: mo.script_language = "Python %d.%d.%d" % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
        mo.session_id = meta.sessionID_S()
        mo.statement_id = meta.statementID()
        mo.node_count = meta.nodeCount()
        mo.node_id = meta.nodeID()
        mo.memory_limit = meta.memoryLimit()
        mo.vm_id = meta.vmID_S()
        mo.input_column_count = meta.inputColumnCount()
        mo.input_column_name = [decodeUTF8(meta.inputColumnName(x)) for x in range(mo.input_column_count)]
        if meta.inputType() == EXACTLY_ONCE: mo.input_type = "SCALAR"
        else: mo.input_type = "SET"
        mo.output_column_count = meta.outputColumnCount()
        if meta.outputType() == EXACTLY_ONCE: mo.output_type = "RETURN"
        else: mo.output_type = "EMIT"
        def ci(x, tbl):
            if tbl == 'input':
                colname = decodeUTF8(self.__meta.inputColumnName(x))
                coltype = self.__meta.inputColumnType(x)
                colprec = self.__meta.inputColumnPrecision(x)
                colscale = self.__meta.inputColumnScale(x)
                colsize = self.__meta.inputColumnSize(x)
                coltn = self.__meta.inputColumnTypeName(x)
            elif tbl == 'output':
                colname = decodeUTF8(self.__meta.outputColumnName(x))
                coltype = self.__meta.outputColumnType(x)
                colprec = self.__meta.outputColumnPrecision(x)
                colscale = self.__meta.outputColumnScale(x)
                colsize = self.__meta.outputColumnSize(x)
                coltn = self.__meta.outputColumnTypeName(x)
            class exacolumn:
                def __init__(self, cn, ct, st, cp, cs, l):
                    self.name = cn
                    self.type = ct
                    self.sql_type = st
                    self.precision = cp
                    self.scale = cs
                    self.length = l
            if coltype == INT32: return exacolumn(colname, int, coltn, colprec, 0, None)
            elif coltype == INT64: return exacolumn(colname, int, coltn, colprec, 0, None)
            elif coltype == DOUBLE: return exacolumn(colname, float, coltn, None, None, None)
            elif coltype == STRING: return exacolumn(colname, unicode, coltn, None, None, colsize)
            elif coltype == BOOLEAN: return exacolumn(colname, bool, coltn, None, None, None)
            elif coltype == NUMERIC and colscale == 0: return exacolumn(colname, int, coltn, colprec, 0, None)
            elif coltype == NUMERIC: return exacolumn(colname, decimal.Decimal, coltn, colprec, colscale, None)
            elif coltype == DATE: return exacolumn(colname, datetime.date, coltn, None, None, None)
            elif coltype == TIMESTAMP: return exacolumn(colname, datetime.datetime, coltn, None, None, None)
            return exacolumn(colname, coltype, coltn, colprec, colscale, colsize)
        mo.input_columns = [ci(x, 'input') for x in range(mo.input_column_count)]
        mo.output_columns = [ci(x, 'output') for x in range(mo.output_column_count)]
        self.meta = mo

    def import_script(self, script):
        modobj = None
        modname = unicode(script)
        code = self.__meta.moduleContent(encodeUTF8(modname))
        msg = self.__meta.checkException()

        if msg: raise ImportError(u"F-UDF-CL-SL-PYTHON-1119: Importing module %s failed: %s" % (modname, msg))
        code = decodeUTF8(code)
        if str(code) in self.__modules:
            print("%%% found code", modname, repr(code))
            modobj = self.__modules[str(code)]
        else:
            print("%%% new code", modname, repr(code), code in self.__modules)
            modobj = imp.new_module(modname)
            modobj.__file__ = "<%s>" % modname
            modobj.__dict__['exa'] = self
            self.__modules[str(code)] = modobj
            try:
                if sys.version_info[0] >= 3:
                    exec(compile(code, script, 'exec'), modobj.__dict__)
                else:
                    exec(compile(code, script, 'exec')) in modobj.__dict__
            except BaseException as err:
                raise create_exception_with_complete_backtrace(
                        "F-UDF-CL-SL-PYTHON-1120",
                        "Importing module %s failed"%modname,
                        sys.exc_info())
        return modobj


    class ConnectionInformation:
        def __init__(self,type,address,user,password):
            self.type = type
            self.address = address
            self.user = user
            self.password = password

        def __str__(self):
            return "{\"type\":\""+self.type+"\",\"address\":\""+self.address+"\",\"user\":\""+self.user+"\",\"password\":<omitted!>}"


    def get_connection(self, name):
        connection_name = unicode(name)
        connectionInfo = self.__meta.connectionInformation(encodeUTF8(connection_name))
        msg = self.__meta.checkException()
        if msg: raise ImportError(u"F-UDF-CL-SL-PYTHON-1121: get_connection for connection name %s failed: %s" % (name, msg))
        return exa.ConnectionInformation(decodeUTF8(connectionInfo.copyKind()), decodeUTF8(connectionInfo.copyAddress()), decodeUTF8(connectionInfo.copyUser()), decodeUTF8(connectionInfo.copyPassword()))


    def redirect_output(self, target = ('192.168.1.1', 5000)):
        import socket
        class activate_remote_output:
            def __init__(self):
                self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.s.connect(target)
                sys.stdout = sys.stderr = self
            def write(self, data): return self.s.sendall(data)
            def close(self): self.s.close()
        activate_remote_output()

exa = exa()

def __pythonvm_wrapped_parse(env):
    try:
        exec(compile(exa.meta.script_code, exa.meta.script_name, 'exec'), env)
    except BaseException as err:
        raise create_exception_with_complete_backtrace(
                "F-UDF-CL-SL-PYTHON-1122",
                "Exception while parsing UDF",
                sys.exc_info())
