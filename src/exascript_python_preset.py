import sys, os

sys.path.append("/usr/lib/python2.7/dist-packages")

from exascript_python import *
import decimal
import datetime
import traceback
import imp

class exa:
    def __init__(self):
        self.__modules = {}
        self.__meta = meta = Metadata()
        class exameta: pass
        mo = exameta()
        mo.database_name = meta.databaseName()
        mo.database_version = meta.databaseVersion()
        mo.script_name = meta.scriptName().decode('utf-8')
        mo.script_schema = meta.scriptSchema().decode('utf-8')
        mo.current_user = meta.currentUser().decode('utf-8')
        mo.scope_user = meta.scopeUser().decode('utf-8')
        mo.current_schema = meta.currentSchema().decode('utf-8')
        mo.scope_user = meta.scopeUser().decode('utf-8')
        mo.script_code = meta.scriptCode().decode('utf-8')
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
        mo.input_column_name = [meta.inputColumnName(x).decode('utf-8') for x in range(mo.input_column_count)]
        if meta.inputType() == EXACTLY_ONCE: mo.input_type = "SCALAR"
        else: mo.input_type = "SET"
        mo.output_column_count = meta.outputColumnCount()
        if meta.outputType() == EXACTLY_ONCE: mo.output_type = "RETURN"
        else: mo.output_type = "EMIT"
        def ci(x, tbl):
            if tbl == 'input':
                colname = self.__meta.inputColumnName(x).decode('utf-8')
                coltype = self.__meta.inputColumnType(x)
                colprec = self.__meta.inputColumnPrecision(x)
                colscale = self.__meta.inputColumnScale(x)
                colsize = self.__meta.inputColumnSize(x)
                coltn = self.__meta.inputColumnTypeName(x)
            elif tbl == 'output':
                colname = self.__meta.outputColumnName(x).decode('utf-8')
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
        modname = unicode(script)
        code = self.__meta.moduleContent(modname.encode('utf-8'))
        msg = self.__meta.checkException()
        if msg: raise ImportError(u"Importing module %s failed: %s" % (modname, msg))
        code = code.decode('utf-8')
        if self.__modules.has_key(code):
            print "%%% found code", modname, repr(code)
            modobj = self.__modules[code]
        else:
            print "%%% new code", modname, repr(code), code in self.__modules
            modobj = imp.new_module(modname)
            modobj.__file__ = "<%s>" % modname
            modobj.__dict__['exa'] = self
            self.__modules[code] = modobj
            try: exec compile(code, script, 'exec') in modobj.__dict__
            except Exception, err:
                raise ImportError(u"Importing module %s failed: %s" % (modname, str(err)))
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
        connectionInfo = self.__meta.connectionInformation(connection_name.encode('utf-8'))
        msg = self.__meta.checkException()
        if msg: raise ImportError(u"get_connection for connection name %s failed: %s" % (name, msg))
        return exa.ConnectionInformation(connectionInfo.copyKind().decode('utf-8'), connectionInfo.copyAddress().decode('utf-8'), connectionInfo.copyUser().decode('utf-8'), connectionInfo.copyPassword().decode('utf-8'))


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

def __pythonvm_wrapped_parse():
    try: exec compile(exa.meta.script_code, exa.meta.script_name, 'exec') in globals()
    except Exception, err:
        errtypel, errobj, backtrace = sys.exc_info()
        if backtrace.tb_next: backtrace = backtrace.tb_next
        err.args = ("".join(traceback.format_exception(errtypel, errobj, backtrace)),)
        raise err
