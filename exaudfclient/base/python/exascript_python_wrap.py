import sys
import os
pyextdataframe_pkg = None

unicode = str
decodeUTF8 = lambda x: x
encodeUTF8 = lambda x: x
long = int
    
if 'LIBPYEXADATAFRAME_DIR' in os.environ:
    path_to_pyexadataframe=os.environ['LIBPYEXADATAFRAME_DIR']
    #print("sys.path append",path_to_pyexadataframe)
    sys.path.append(path_to_pyexadataframe)
else:
    path_to_pyexadataframe="/exaudf/external/exaudfclient_base+/python/python3"
    #print("sys.path append",path_to_pyexadataframe)
    sys.path.append(path_to_pyexadataframe)



class exaiter(object):
    def __init__(self, meta, inp, out):
        self.__meta = meta
        self.__inp = inp
        self.__out = out
        self.__intype = self.__meta.inputType()
        incount = self.__meta.inputColumnCount()
        data = {}
        self.__cache = [None]*incount
        self.__finished = False
        self.__dataframe_finished = False
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
        self.__incoltypes = []
        for col in range(self.__meta.inputColumnCount()):
            self.__incoltypes.append(self.__meta.inputColumnType(col))
        self.__incolnames = []
        for col in range(self.__meta.inputColumnCount()):
            self.__incolnames.append(decodeUTF8(self.__meta.inputColumnName(col)))
        for col in range(incount):
            colname = self.__incolnames[col]
            if self.__incoltypes[col] == DOUBLE:
                data[colname] = rd(inp.getDouble, inp.wasNull, col)
            elif self.__incoltypes[col] == STRING:
                data[colname] = rd(inp.getString, inp.wasNull, col, lambda x: decodeUTF8(x))
            elif self.__incoltypes[col] == INT32:
                data[colname] = rd(inp.getInt32, inp.wasNull, col)
            elif self.__incoltypes[col] == INT64:
                data[colname] = rd(inp.getInt64, inp.wasNull, col)
            elif self.__incoltypes[col] == NUMERIC:
                if self.__meta.inputColumnScale(col) == 0:
                    data[colname] = rd(inp.getNumeric, inp.wasNull, col, lambda x: int(str(x)))
                else: data[colname] = rd(inp.getNumeric, inp.wasNull, col, lambda x: decimal.Decimal(str(x)))
            elif self.__incoltypes[col] == DATE:
                data[colname] = rd(inp.getDate, inp.wasNull, col, convert_date)
            elif self.__incoltypes[col] == TIMESTAMP:
                data[colname] = rd(inp.getTimestamp, inp.wasNull, col, convert_timestamp)
            elif self.__incoltypes[col] == BOOLEAN:
                data[colname] = rd(inp.getBoolean, inp.wasNull, col)
            data[col] = data[colname]
        self.__outcoltypes = []
        for col in range(self.__meta.outputColumnCount()):
            self.__outcoltypes.append(self.__meta.outputColumnType(col))
        self.__data = data
    def __getitem__(self, key):
        if self.__finished:
            raise RuntimeError("E-UDF-CL-SL-PYTHON-1081: Iteration finished")
        if key not in self.__data:
            key = unicode(key)
            if key not in self.__data:
                raise RuntimeError(u"E-UDF-CL-SL-PYTHON-1082: Column with name '%s' does not exist" % key)
        ret, null = self.__data[key]()
        msg = self.__inp.checkException()
        if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1083: "+msg)
        if null: return None
        return ret
    def __getattr__(self, key):
        if self.__finished:
            raise RuntimeError("E-UDF-CL-SL-PYTHON-1084: Iteration finished")
        if key not in self.__data:
            key = unicode(key)
            if key not in self.__data:
                raise RuntimeError(u"E-UDF-CL-SL-PYTHON-1085: Iterator has no object with name '%s'" % key)
        ret, null = self.__data[key]()
        msg = self.__inp.checkException()
        if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1086: "+msg)
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
        if len(output) == 1 and output[0].__class__.__name__ == 'DataFrame':
            global pyextdataframe_pkg
            if pyextdataframe_pkg is None:
                import pyextdataframe
                pyextdataframe_pkg = pyextdataframe

            v = output[0]
            if v.shape[0] == 0:
                raise RuntimeError("E-UDF-CL-SL-PYTHON-1087: emit DataFrame is empty")
            if v.shape[1] != len(self.__outcoltypes):
                exp_num_out = len(self.__outcoltypes)
                raise TypeError("E-UDF-CL-SL-PYTHON-1088: emit() takes exactly %d argument%s (%d given)" % (exp_num_out, 's' if exp_num_out > 1 else '', v.shape[1]))
            pyextdataframe_pkg.emit_dataframe(self, v)
            return
        if len(output) != len(self.__outcoltypes):
            if len(self.__outcoltypes) > 1:
                raise TypeError("E-UDF-CL-SL-PYTHON-1089: emit() takes exactly %d arguments (%d given)" % (len(self.__outcoltypes), len(output)))
            else: raise TypeError("E-UDF-CL-SL-PYTHON-1090: emit() takes exactly %d argument (%d given)" % (len(self.__outcoltypes), len(output)))
        for v in output:
            if v == None: self.__out.setNull(k)
            elif type(v) in (int, long):
                if self.__outcoltypes[k] == INT32: self.__out.setInt32(k, int(v))
                elif self.__outcoltypes[k] == INT64: self.__out.setInt64(k, int(v))
                elif self.__outcoltypes[k] == NUMERIC: self.__out.setNumeric(k, str(int(v)))
                elif self.__outcoltypes[k] == DOUBLE: self.__out.setDouble(k, float(v))
                else:
                    raise RuntimeError(u"E-UDF-CL-SL-PYTHON-1091: emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
            elif type(v) == float:
                if self.__outcoltypes[k] == DOUBLE: self.__out.setDouble(k, float(v))
                elif self.__outcoltypes[k] == INT32: self.__out.setInt32(k, int(v))
                elif self.__outcoltypes[k] == INT64: self.__out.setInt64(k, int(v))
                elif self.__outcoltypes[k] == NUMERIC: self.__out.setInt64(k, str(v))
                else:
                    raise RuntimeError(u"E-UDF-CL-SL-PYTHON-1092: emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
            elif type(v) == bool:
                if self.__outcoltypes[k] != BOOLEAN:
                    raise RuntimeError(u"E-UDF-CL-SL-PYTHON-1093: emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setBoolean(k, bool(v))
            elif type(v) in (str, unicode):
                v = encodeUTF8(v)
                vl = len(v)
                if self.__outcoltypes[k] != STRING:
                    raise RuntimeError(u"E-UDF-CL-SL-PYTHON-1094: emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setString(k, v, vl)
            elif type(v) == decimal.Decimal:
                if self.__outcoltypes[k] == NUMERIC: self.__out.setNumeric(k, str(v))
                elif self.__outcoltypes[k] == INT32: self.__out.setInt32(k, int(v))
                elif self.__outcoltypes[k] == INT64: self.__out.setInt64(k, int(v))
                elif self.__outcoltypes[k] == DOUBLE: self.__out.setDouble(k, float(v))
                else:
                    raise RuntimeError("E-UDF-CL-SL-PYTHON-1095: emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
            elif type(v) == datetime.date:
                if self.__outcoltypes[k] != DATE:
                    raise RuntimeError("E-UDF-CL-SL-PYTHON-1096: emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setDate(k, v.isoformat())
            elif type(v) == datetime.datetime:
                if self.__outcoltypes[k] != TIMESTAMP:
                    raise RuntimeError("E-UDF-CL-SL-PYTHON-1097: emit column '%s' is of type %s but data given have type %s" \
                            % (decodeUTF8(self.__meta.outputColumnName(k)), type_names.get(self.__outcoltypes[k], 'UNKONWN'), str(type(v))))
                self.__out.setTimestamp(k, v.isoformat(' '))
            else: raise RuntimeError("E-UDF-CL-SL-PYTHON-1098: data type %s is not supported" % str(type(v)))
            msg = self.__out.checkException()
            if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1099: "+msg)
            k += 1
        ret = self.__out.next()
        msg = self.__out.checkException()
        if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1100: "+msg)
        if ret != True: raise RuntimeError("F-UDF-CL-SL-PYTHON-1101: Internal error on emiting row")
    def next(self, reset = False):
        self.__cache = [None] * len(self.__cache)
        if reset:
            self.__inp.reset()
            self.__finished = False
            val = True
        elif self.__finished: return False
        else: val = self.__inp.next()
        msg = self.__inp.checkException()
        if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1102: "+msg)
        if not val:
            self.__finished = True
        return val
    def get_dataframe(self, num_rows=1, start_col=0):
        global pyextdataframe_pkg
        if pyextdataframe_pkg is None:
            import pyextdataframe
            pyextdataframe_pkg = pyextdataframe

        if not (num_rows == "all" or (type(num_rows) in (int, long) and num_rows > 0)):
            raise RuntimeError("E-UDF-CL-SL-PYTHON-1103: get_dataframe() parameter 'num_rows' must be 'all' or an integer > 0")
        if (type(start_col) not in (int, long) or start_col < 0):
            raise RuntimeError("E-UDF-CL-SL-PYTHON-1104: get_dataframe() parameter 'start_col' must be an integer >= 0")
        if (start_col > len(self.__incolnames)):
            raise RuntimeError("E-UDF-CL-SL-PYTHON-1105: get_dataframe() parameter 'start_col' is %d, but there are only %d input columns" % (start_col, len(self.__incolnames)))
        if num_rows == "all":
            num_rows = sys.maxsize
        if self.__dataframe_finished:
            # Exception after None already returned
            raise RuntimeError("E-UDF-CL-SL-PYTHON-1106: Iteration finished")
        elif self.__finished:
            # Return None the first time there is no data
            self.__dataframe_finished = True
            return None
        self.__cache = [None] * len(self.__cache)
        df = pyextdataframe_pkg.get_dataframe(self, num_rows, start_col)
        return df 
    def reset(self):
        self.__dataframe_finished = False
        return self.next(reset = True)
    def size(self):
        return self.__inp.rowsInGroup()

def __disallowed_function(*args, **kw):
    raise RuntimeError("F-UDF-CL-SL-PYTHON-1107: next(), reset() and emit() functions are not allowed in scalar context")

def __pythonvm_wrapped_cleanup():
    cleanupfunc = None
    try: cleanupfunc = globals()['cleanup']
    except: raise RuntimeError("F-UDF-CL-SL-PYTHON-1108: function 'cleanup' is not defined")
    try:
        cleanupfunc()
    except BaseException as err:
        raise create_exception_with_complete_backtrace(
                "F-UDF-CL-SL-PYTHON-1109",
                "Exception during cleanup",
                sys.exc_info())

def __pythonvm_wrapped_run():
    runfunc = None
    try: runfunc = globals()['run']
    except: raise RuntimeError("F-UDF-CL-SL-PYTHON-1110: function 'run' is not defined")
    inp = TableIterator(); msg = inp.checkException();
    if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1111: "+msg)
    out = ResultHandler(inp); msg = out.checkException();
    if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1112: "+msg)
    meta = Metadata(); msg = meta.checkException();
    if msg: raise RuntimeError("F-UDF-CL-SL-PYTHON-1113: "+msg)
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
    except BaseException as err:
        raise create_exception_with_complete_backtrace(
                "F-UDF-CL-SL-PYTHON-1114",
                "Exception during run",
                sys.exc_info())


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
            try:
                return fn(imp_spec)
            except BaseException as err:
                raise create_exception_with_complete_backtrace(
                        "F-UDF-CL-SL-PYTHON-1115",
                        "Exception during singleCall %s"%fn.__name__,
                        sys.exc_info())
        elif "generate_sql_for_export_spec" in globals() and fn == generate_sql_for_export_spec:
            exp_spec = __ExportSpecification(arg)
            if exp_spec.connection:
                exp_spec._setConnectionInformation(__ConnectionInformation(exp_spec.connection))
            try:
                return fn(exp_spec)
            except BaseException as err:
                raise create_exception_with_complete_backtrace(
                        "F-UDF-CL-SL-PYTHON-1116",
                        "Exception during singleCall %s"%fn.__name__,
                        sys.exc_info())
        else:
            raise RuntimeError("F-UDF-CL-SL-PYTHON-1117: Unknown single call function: "+str(fn))
    else:
        try:
            return fn()
        except BaseException as err:
            raise create_exception_with_complete_backtrace(
                    "F-UDF-CL-SL-PYTHON-1118",
                    "Exception during singleCall %s"%fn.__name__,
                    sys.exc_info())

