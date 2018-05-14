import zmq, decimal, datetime, imp
import zmqcontainer_pb2 as z
MAX_DATASIZE = 6000000

class comm:
    """ General communication function

        Connects to EXASolution with given client_name and works as
        callable object, to send requests and receive
        responses. Request and response objects are cleared
        automatically on each call, after sending request and before
        receiving response.
    """
    
    def __init__(self, client_name):
        """ This function estabilish the connection to EXASolution """
        self.z = z
        self.connection_id = 0
        self.req = z.exascript_request()
        self.rep = z.exascript_response()
        self.client_name = client_name
        self.zcontext = zmq.Context()
        self.zsocket = self.zcontext.socket(zmq.REQ)
        if self.client_name.startswith('tcp://'):
            self.zsocket.bind(self.client_name)
        elif self.client_name.startswith('unix://'):
            self.zsocket.connect(self.client_name)
        else: raise RuntimeError("Unsupported protocol, supported are only tcp:// and unix://")

    
    def __call__(self, req_type, rep_type = None, req = None, rep = None):
        """ Communication functionality

            Send the request of given type and receive response of
            given type(s). If no req or rep is given, use internal
            objects.
        """
        z = self.z
        if req == None: req = self.req
        if rep == None: rep = self.rep
        req.type = req_type
        req.connection_id = self.connection_id
        self.zsocket.send(req.SerializeToString())
        req.Clear(); rep.Clear()
        n = lambda x: x != None and z._MESSAGE_TYPE.values[x].name or "None"
        comm_desc = "COMM %s - %s->%s:" % \
                    (self.connection_id,
                     n(req_type),
                     " ".join(tuple(n(x) for x in type(rep_type) == tuple and rep_type or (rep_type,))))
        if rep_type is None:
            print comm_desc, "(without response)"
            return None
        rep.ParseFromString(self.zsocket.recv())
        if self.rep.type == z.MT_CLOSE:
            print comm_desc, "-> CLOSE:", rep.close.exception_message
            raise RuntimeError("Error: " + rep.close.exception_message)
        if type(rep_type) in (list, tuple):
            if rep.type not in rep_type:
                print comm_desc, "-> unexpected response:", repr(rep_type)
                raise RuntimeError("Unexpected message: " + repr(rep.type) + " not in " + repr(rep_type))
        elif rep.type != rep_type:
            print comm_desc, "-> unexpected response:", repr(rep_type)
            raise RuntimeError("Unexpected message: " + repr(rep.type) + " not in " + repr(rep_type))
        if self.connection_id != 0 and rep.connection_id != self.connection_id:
            print comm_desc, "-> wrong connection:", self.connection_id, "!=", rep.connection_id
            raise RuntimeError("Received wrong connection ID")
        #print comm_desc, n(rep.type), "OK"

class exa:
    """ "exa" object

        Initialy parses the meta information and present it to the
        user exactly in the same way as in EXASolution internal UDF.
    """

    def __init__(self, comm):
        self._z = comm.z
        self._comm = comm
        self._comm.req.client.client_name = comm.client_name
        self._comm.req.client.meta_info = ""
        self._comm(z.MT_CLIENT, z.MT_INFO)
        self._comm.connection_id = self._comm.rep.connection_id

        class exameta: pass
        mo = exameta()
        mo.database_name    = self._comm.rep.info.database_name
        mo.database_version = self._comm.rep.info.database_version
        mo.script_name      = self._comm.rep.info.script_name
        mo.script_schema    = self._comm.rep.info.script_schema
        mo.current_user     = self._comm.rep.info.current_user
        mo.current_schema   = self._comm.rep.info.current_schema
        mo.scope_user       = self._comm.rep.info.scope_user
        mo.script_code      = self._comm.rep.info.source_code
        mo.script_language  = "Python"
        mo.session_id       = self._comm.rep.info.session_id
        mo.statement_id     = self._comm.rep.info.statement_id
        mo.node_count       = self._comm.rep.info.node_count
        mo.node_id          = self._comm.rep.info.node_id
        mo.vm_id            = self._comm.rep.info.vm_id
        self._meta_info     = self._comm.rep.info.meta_info

        cfg_input_columns = []
        cfg_output_columns = []
        self._comm(z.MT_META, z.MT_META)

        self._single_call_mode = self._comm.rep.meta.single_call_mode

        if self._comm.rep.meta.input_iter_type == z.PB_EXACTLY_ONCE:
            cfg_input_type = 'SCALAR'
        elif self._comm.rep.meta.input_iter_type == z.PB_MULTIPLE:
            cfg_input_type = 'SET'
        else: raise RuntimeError("Unkown input iteration type")

        if self._comm.rep.meta.output_iter_type == z.PB_EXACTLY_ONCE:
            cfg_output_type = 'RETURN'
        elif self._comm.rep.meta.output_iter_type == z.PB_MULTIPLE:
            cfg_output_type = 'EMIT'
        else: raise RuntimeError("Unkown output iteration type")

        self._input_type  = cfg_input_type
        self._output_type = cfg_output_type

        for o, i in [(cfg_input_columns, self._comm.rep.meta.input_columns),
                     (cfg_output_columns, self._comm.rep.meta.output_columns)]:
            for col in i:
                info = {
                    'name'      : col.name,
                    'pbtype'    : col.type,
                    'sqltype'   : col.type_name,
                    'length'    : col.size,
                    'precision' : col.precision,
                    'scale'     : col.scale,
                }
                if   col.type == z.PB_DOUBLE:    info['type'] = float
                elif col.type == z.PB_INT32:     info['type'] = int
                elif col.type == z.PB_INT64:     info['type'] = long
                elif col.type == z.PB_NUMERIC:   info['type'] = decimal.Decimal
                elif col.type == z.PB_TIMESTAMP: info['type'] = datetime.datetime
                elif col.type == z.PB_DATE:      info['type'] = datetime.date
                elif col.type == z.PB_STRING:    info['type'] = unicode
                elif col.type == z.PB_BOOLEAN:   info['type'] = bool
                else: raise RuntimeError("Received unknown column type")
                if col.type in (z.PB_DOUBLE, z.PB_INT32, z.PB_INT64, z.PB_NUMERIC):
                    info['length']= None
                o.append(info)
        self._input_columns  = cfg_input_columns
        self._output_columns = cfg_output_columns

        mo.input_type          = cfg_input_type
        mo.output_type         = cfg_output_type
        mo.input_column_count  = len(cfg_input_columns)
        mo.output_column_count = len(cfg_output_columns)
        class exacolumn:
            def __init__(self, d):
                self.name      = d['name']
                self.type      = d['type']
                self.sql_type  = d['sqltype']
                self.precision = d.get('precision', None)
                self.scale     = d.get('scale', None)
                self.length    = d.get('length', None)
            def __repr__(self):
                return "<col n:%s t:%s T:%s p:%s s:%s l:%s>" % tuple(repr(x) for x in (
                    self.name, self.type, self.sql_type, self.precision, self.scale, self.length))
        mo.input_columns  = [exacolumn(a) for a in cfg_input_columns]
        mo.output_columns = [exacolumn(a) for a in cfg_output_columns]
        self.meta = mo
        self.__modules = {}

    def import_script(self, script):
        """ Import script from the EXASolution database """
        z = self._z
        getattr(self._comm.req, 'import').script_name = script
        try:
            self._comm(z.MT_IMPORT, z.MT_IMPORT)
            if getattr(self._comm.rep, 'import').exception_message != '':
                message = getattr(self._comm.rep, 'import').exception_message
                raise RuntimeError(message)
            code = getattr(self._comm.rep, 'import').source_code
        except Exception, err:
            raise ImportError(u'importing script %s failed: %s' % (script, str(err)))
        print "IMPORT", self._comm.connection_id, repr(code), "cache", repr(self), repr(self.__modules)
        if self.__modules.has_key(code):            
            return self.__modules[code]
        obj = imp.new_module(script)
        obj.__file__ = '<%s>' % script
        obj.__dict__['exa'] = self
        self.__modules[code] = obj
        try: exec compile(code, script, 'exec') in obj.__dict__
        except Exception, err:
            raise ImportError(u'importing module %s failed: %s' % (script, str(err)))
        return obj

class exaiter:
    """ Data iterator

        Allows to iterate through the input data. Initially creates
        functions to properly convert input data to Python objects.
    """

    def __init__(self, exa):
        comm = exa._comm; z = comm.z
        self.__comm     = comm
        self.__z        = z
        self.__input    = z.exascript_response()
        self.__output   = z.exascript_request()
        self.__row_num  = 0
        self.__finished = False
        self.__colnums  = {}

        # define the read_row function, which reads a full row from
        # received data
        def convert_date(x):
            val = datetime.datetime.strptime(x, "%Y-%m-%d")
            return datetime.date(val.year, val.month, val.day)
        def convert_timestamp(x):
            return datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f")
        self.__input_offsets    = {}
        input_types      = []
        input_converters = []
        readfuns         = []
        colnum           = 0
        for col in exa._input_columns:
            converter = col['type']
            if   col['pbtype'] == z.PB_DOUBLE:
                input_types.append(('double', colnum,
                   lambda x: self.__input.next.table.data_double[x]))
            elif col['pbtype'] == z.PB_INT32:
                input_types.append(('int32', colnum,
                   lambda x: self.__input.next.table.data_int32[x]))
            elif col['pbtype'] == z.PB_INT64:
                input_types.append(('int64', colnum,
                   lambda x: self.__input.next.table.data_int64[x]))
            elif col['pbtype'] == z.PB_BOOLEAN:
                input_types.append(('bool', colnum,
                   lambda x: self.__input.next.table.data_bool[x]))
            elif col['pbtype'] in (z.PB_NUMERIC, z.PB_TIMESTAMP, z.PB_DATE, z.PB_STRING):
                if col['pbtype'] == z.PB_DATE: converter = convert_date
                elif col['pbtype'] == z.PB_TIMESTAMP: converter = convert_timestamp
                input_types.append(('string', colnum, lambda x: self.__input.next.table.data_string[x]))
            else: raise RuntimeError("Unknown type")
            self.__colnums[col['name']] = colnum
            self.__colnums[unicode(colnum)] = colnum
            colnum += 1
            input_converters.append(converter)
        def read_row():
            self.__row = []
            for ct, cn, cr in input_types:
                offset = self.__input_offsets.get('null', 0)
                self.__input_offsets['null'] = offset + 1
                if offset >= len(self.__input.next.table.data_nulls):
                    table = self.__input.next.table
                if self.__input.next.table.data_nulls[offset]:
                    self.__row.append(None)
                else:
                    offset = self.__input_offsets.get(ct, 0)
                    self.__input_offsets[ct] = offset + 1
                    self.__row.append(input_converters[cn](cr(offset)))
        self.__read_row = read_row

        # define the write_row function, which writes a full row to
        # the output buffer
        writefuns = []
        def get_data(x, c, n, t):
            if type(x) not in t:
                raise RuntimeError("Column is of type %s but data given have type %s" % (n, str(type(x))))
            return c(x)
        for col in exa._output_columns:
            if col['pbtype'] == z.PB_DOUBLE:
                writefuns.append(lambda x: self.__output.emit.table.data_double.append(get_data(x, float, 'float', (float, int, long, decimal.Decimal))) or 12)
            elif col['pbtype'] == z.PB_INT32:
                writefuns.append(lambda x: self.__output.emit.table.data_int32.append(get_data(x, int, 'int', (float, int, long, decimal.Decimal))) or 4)
            elif col['pbtype'] == z.PB_INT64:
                writefuns.append(lambda x: self.__output.emit.table.data_int64.append(get_data(x, long, 'long', (float, int, long, decimal.Decimal))) or 8)
            elif col['pbtype'] in (z.PB_NUMERIC, z.PB_STRING):
                def wf(x):
                    d = unicode(x)
                    self.__output.emit.table.data_string.append(d)
                    return len(d)
                writefuns.append(wf)
            elif col['pbtype'] == z.PB_TIMESTAMP:
                def wf(x):
                    d = x.isoformat(' ')
                    self.__output.emit.table.data_string.append(d)
                    return len(d)
                writefuns.append(wf)
            elif col['pbtype'] == z.PB_DATE:
                def wf(x):
                    d = x.isoformat()
                    self.__output.emit.table.data_string.append(d)
                    return len(d)
                writefuns.append(wf)
            elif col['pbtype'] == z.PB_BOOLEAN:
                writefuns.append(lambda x: self.__output.emit.table.data_bool.append(get_data(x, bool, 'bool', (bool,))) or 1)
        def write_row(row):
            for d, f in zip(row, writefuns):
                if d != None:
                    self.__written_bytes += f(d)
                    self.__output.emit.table.data_nulls.append(False)
                else: self.__output.emit.table.data_nulls.append(True)
                self.__written_bytes += 1
            self.__output.emit.table.rows += 1
        self.__written_bytes = 0

        self.__write_row = write_row
        self.__output_columns_count = len(exa._output_columns)

        # read first block
        #self.next()

    def __getitem__(self, key):
        # allows to access columns with this syntax: it["colname"]
        key = unicode(key)
        if key not in self.__colnums:
            raise RuntimeError(u"Column with name '%s' does not exist" % key)
        if self.__finished: raise RuntimeError("Iteration finished")
        return self.__row[self.__colnums[key]]

    def __getattr__(self, key):
        # allows to access columns with this symtax: it.colname
        key = unicode(key)
        if key not in self.__colnums:
            raise RuntimeError(u"Column with name '%s' does not exist" % key)
        if self.__finished: raise RuntimeError("Iteration finished")
        return self.__row[self.__colnums[key]]

    def emit(self, *output):
        # emits one row
        # if size of all emmited rows in the buffer is larger then
        # MAX_DATASIZE, then flush the buffer.
        comm = self.__comm; z = self.__z
        if len(output) == self.__output_columns_count:
            self.__write_row(output)
            limit = MAX_DATASIZE
        elif len(output) == 0: limit = 0
        else: raise RuntimeError("Emited wrong number of columns")
        if self.__written_bytes > limit:
            print "EMIT", comm.connection_id, "bytes sent:", self.__written_bytes
            self.__output.emit.table.rows_in_group = 0
            comm(z.MT_EMIT, z.MT_EMIT, req = self.__output)
            self.__written_bytes = 0
        return None

    def next(self, reset = False, first = False):
        # reads next row
        # if no rows available, fetch next block
        # if still no rows available, return False to indicate, that
        # no more rows available
        # if reset is True, fetch first block of current group and
        # beginn with first row
        # returns True to indicate, that next row was fetched.
        comm = self.__comm; z = self.__z
        if first or reset: self.__finished = False
        elif self.__finished: return False
        if self.__row_num == 0 or self.__row_num >= self.__input.next.table.rows:
            if reset: comm(z.MT_RESET, (z.MT_RESET, z.MT_DONE), rep = self.__input)
            else: comm(z.MT_NEXT, (z.MT_NEXT, z.MT_DONE), rep = self.__input)
            self.__row_num = 0
            for ot in self.__input_offsets.keys():
                self.__input_offsets[ot] = 0
            if self.__input.type == z.MT_DONE:
                print "ITER", comm.connection_id, "read next row", "(finished)"
                self.__finished = True
                return False
            else:
                print "ITER", comm.connection_id, "read next row", self.__input.next.table.rows
        self.__read_row()
        self.__row_num += 1
        return True

    def reset(self):
        return self.next(reset = True)

    def size(self):
        # returns number of rows in current group
        if self.__finished: return 0
        return self.__input.next.table.rows_in_group

def disallowed_function(*args, **kw):
    raise RuntimeError("next(), reset() and emit() functions are not allowed in scalar context")

__all__ = ['comm', 'exaiter', 'disallowed_function']
