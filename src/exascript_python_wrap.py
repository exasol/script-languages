import sys
isPython3 = False
if sys.version_info[0] == 3:
    unicode = str
    decodeUTF8 = lambda x: x
    long = int
    isPython3 = True
else:
    decodeUTF8 = lambda x: x.decode('utf-8')

class exaiter(object):
    def __init__(self, meta, inp, out):
        self.__meta = meta
        self.__inp = inp
        self.__out = out
        incount = self.__meta.inputColumnCount()
        data = {}
        self.__cache = [None]*incount
        self.__finished = False
        def rd(get, null, col, postfun = None):
            if postfun == None:
                newget = lambda: (get(col), null())
            else:
                def newget():
                    v = get(col)
                    n = null()
                    if n: return (v, True)
                    return (postfun(v), False)
            def resget():
                val = self.__cache[col]
                if val == None:
                    val = self.__cache[col] = newget()
                return val
            return resget
        def convert_date(x):
            val = datetime.datetime.strptime(x, "%Y-%m-%d")
            return datetime.date(val.year, val.month, val.day)
        def convert_timestamp(x):
            return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
        for col in range(incount):
            colname = decodeUTF8(self.__meta.inputColumnName(col))
            if self.__meta.inputColumnType(col) == DOUBLE:
                data[colname] = rd(inp.getDouble, inp.wasNull, col)
            elif self.__meta.inputColumnType(col) == STRING:
                data[colname] = rd(inp.getString, inp.wasNull, col, lambda x: decodeUTF8(x))
            elif self.__meta.inputColumnType(col) == INT32:
                data[colname] = rd(inp.getInt32, inp.wasNull, col)
            elif self.__meta.inputColumnType(col) == INT64:
                data[colname] = rd(inp.getInt64, inp.wasNull, col)
            elif self.__meta.inputColumnType(col) == NUMERIC:
                if self.__meta.inputColumnScale(col) == 0:
                    data[colname] = rd(inp.getNumeric, inp.wasNull, col, lambda x: int(str(x)))
                else: data[colname] = rd(inp.getNumeric, inp.wasNull, col, lambda x: decimal.Decimal(str(x)))
            elif self.__meta.inputColumnType(col) == DATE:
                data[colname] = rd(inp.getDate, inp.wasNull, col, convert_date)
            elif self.__meta.inputColumnType(col) == TIMESTAMP:
                data[colname] = rd(inp.getTimestamp, inp.wasNull, col, convert_timestamp)
            elif self.__meta.inputColumnType(col) == BOOLEAN:
                data[colname] = rd(inp.getBoolean, inp.wasNull, col)
            data[col] = data[colname]
        self.__outcoltypes = []
        for col in range(self.__meta.outputColumnCount()):
            self.__outcoltypes.append(self.__meta.outputColumnType(col))
        self.__data = data
    def __getitem__(self, key):
        if self.__finished:
            raise RuntimeError("Iteration finished")
        if key not in self.__data:
            key = unicode(key)
            if key not in self.__data:
                raise RuntimeError(u"Column with name '%s' does not exist" % key)
        ret, null = self.__data[key]()
        msg = self.__inp.checkException()
        if msg: raise RuntimeError(msg)
        if null: return None
        return ret
    def __getattr__(self, key):
        if self.__finished:
            raise RuntimeError("Iteration finished")
        if key not in self.__data:
            key = unicode(key)
            if key not in self.__data:
                raise RuntimeError(u"Iterator has no object with name '%s'" % key)
        ret, null = self.__data[key]()
        msg = self.__inp.checkException()
        if msg: raise RuntimeError(msg)
        if null: return None
        return ret
    def emit(self, *output):
        k = 0
        type_names = {
                DOUBLE: "float",
                BOOLEAN: "bool",
                INT32: "int",
                INT64: "long",
                STRING: "unicode",
                NUMERIC: "decimal.Decimal",
                DATE: "datetime.date",
                TIMESTAMP: "datetime.datetime" }
        if len(output) != len(self.__outcoltypes):
            if len(self.__outcoltypes) > 1:
                raise TypeError("emit() takes exactly %d arguments (%d given)" % (len(self.__outcoltypes), len(output)))
            else: raise TypeError("emit() takes exactly %d argument (%d given)" % (len(self.__outcoltypes), len(output)))
        for v in output:
            if v == None: self.__out.setNull(k)
            elif type(v) in (int, long):
                if self.__outcoltypes[k] == INT32: self.__out.setInt32(k, int(v))
                elif self.__outcoltypes[k] == INT64: self.__out.setInt64(k, int(v))
                elif self.__outcoltypes[k] == NUMERIC: self.__out.setNumeric(k, str(int(v)))
                elif self.__outcoltypes[k] == DOUBLE: self.__out.setDouble(k, float(v))
                else:
                    raise RuntimeError(u"emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
            elif type(v) == float:
                if self.__outcoltypes[k] == DOUBLE: self.__out.setDouble(k, float(v))
                elif self.__outcoltypes[k] == INT32: self.__out.setInt32(k, int(v))
                elif self.__outcoltypes[k] == INT64: self.__out.setInt64(k, int(v))
                elif self.__outcoltypes[k] == NUMERIC: self.__out.setInt64(k, str(v))
                else:
                    raise RuntimeError(u"emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
            elif type(v) == bool:
                if self.__outcoltypes[k] != BOOLEAN:
                    raise RuntimeError(u"emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setBoolean(k, bool(v))
            elif type(v) in (str, unicode):
                vl = len(v)
                if not isPython3 and type(v) == unicode: v = v.encode('utf-8')
                if self.__outcoltypes[k] != STRING:
                    raise RuntimeError(u"emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setString(k, v, vl)
            elif type(v) == decimal.Decimal:
                if self.__outcoltypes[k] == NUMERIC: self.__out.setNumeric(k, str(v))
                elif self.__outcoltypes[k] == INT32: self.__out.setInt32(k, int(v))
                elif self.__outcoltypes[k] == INT64: self.__out.setInt64(k, int(v))
                elif self.__outcoltypes[k] == DOUBLE: self.__out.setDouble(k, float(v))
                else:
                    raise RuntimeError("emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
            elif type(v) == datetime.date:
                if self.__outcoltypes[k] != DATE:
                    raise RuntimeError("emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setDate(k, v.isoformat())
            elif type(v) == datetime.datetime:
                if self.__outcoltypes[k] != TIMESTAMP:
                    raise RuntimeError("emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setTimestamp(k, v.isoformat(' '))
            else: raise RuntimeError("data type %s is not supported" % str(type(v)))
            msg = self.__out.checkException()
            if msg: raise RuntimeError(msg)
            k += 1
        ret = self.__out.next()
        msg = self.__out.checkException()
        if msg: raise RuntimeError(msg)
        if ret != True: raise RuntimeError("Internal error on emiting row")
    def next(self, reset = False):
        self.__cache = [None] * len(self.__cache)
        if reset:
            self.__inp.reset()
            self.__finished = False
            val = True
        elif self.__finished: return False
        else: val = self.__inp.next()
        msg = self.__inp.checkException()
        if msg: raise RuntimeError(msg)
        if not val:
            self.__finished = True
        return val
    def reset(self):
        return self.next(reset = True)
    def size(self):
        return self.__inp.rowsInGroup()

def __disallowed_function(*args, **kw):
    raise RuntimeError("next(), reset() and emit() functions are not allowed in scalar context")

def __pythonvm_wrapped_run():
    runfunc = None
    try: runfunc = run
    except: raise RuntimeError("function 'run' is not defined")
    inp = TableIterator(); msg = inp.checkException();
    if msg: raise RuntimeError(msg)
    out = ResultHandler(inp); msg = out.checkException();
    if msg: raise RuntimeError(msg)
    meta = Metadata(); msg = meta.checkException();
    if msg: raise RuntimeError(msg)
    try:
        iter = exaiter(meta, inp, out); iter_next = iter.next; iter_emit = iter.emit
        if meta.outputType() == EXACTLY_ONCE:
            iter.emit = __disallowed_function
        if meta.inputType() == EXACTLY_ONCE:
            iter.next = iter.reset = __disallowed_function
        if meta.inputType() == MULTIPLE:
            if meta.outputType() == EXACTLY_ONCE: iter_emit(runfunc(iter))
            else: runfunc(iter)
        else:
            if meta.outputType() == EXACTLY_ONCE:
                while(True):
                    iter_emit(runfunc(iter))
                    if not iter_next(): break
            else:
                while(True):
                    runfunc(iter)
                    if not iter_next(): break
        out.flush()
    except Exception as err:
        errtypel, errobj, backtrace = sys.exc_info()
        if backtrace.tb_next: backtrace = backtrace.tb_next
        err.args = ("".join(traceback.format_exception(errtypel, errobj, backtrace)),)
        raise err



class __ImportSpecification:
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

    def _setConnectionInformation(self,value):
        self.d["connection"] = value

    def __getitem__(self, key):
        return self.d[key]


class __ExportSpecification:
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

    def _setConnectionInformation(self,value):
        self.d["connection"] = value

    def __getitem__(self, key):
        return self.d[key]


class __ConnectionInformation:
    def __init__(self, d):
        self.d = d

    def __getattr__(self, key):
        return self.d[key]

    def __getitem__(self, key):
        return self.d[key]


def __pythonvm_wrapped_singleCall(fn,arg=None):
    if arg:
        if "generate_sql_for_import_spec" in globals() and fn == generate_sql_for_import_spec:
            imp_spec = __ImportSpecification(arg)
            if imp_spec.connection:
                imp_spec._setConnectionInformation(__ConnectionInformation(imp_spec.connection))
            return fn(imp_spec)
        elif "generate_sql_for_export_spec" in globals() and fn == generate_sql_for_export_spec:
            exp_spec = __ExportSpecification(arg)
            if exp_spec.connection:
                exp_spec._setConnectionInformation(__ConnectionInformation(exp_spec.connection))
            return fn(exp_spec)
        else:
            raise RuntimeError("Unknown single call function: "+str(fn))
    else:
        return fn()
