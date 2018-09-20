import zmq, sys, locale, csv, time, threading
from zmqcontainer_pb2 import *
locale.setlocale(locale.LC_ALL, "")

SWIG_MAX_VAR_DATASIZE = 6000000

if len(sys.argv) != 5:
    sys.stderr.write("Usage: %s <target> <csvinput> <csvoutput> <script>\n" % sys.argv[0])
    sys.stderr.write("""
Target can be:
    unix:/file/path.socket        - path to UNIX socket of ZMQContainerClient
    redir:127.0.0.1:2000          - connection string to redirector
    tcp:127.0.0.1:2000            - host and port of running ZMQContainerClient
""")
    
    sys.exit(1)

traffic = {'send': 0, 'recv': 0}
lp_count = {'global': 0, 'next': 0, 'emit': 0, 'wait': 0}
def lp(typ):
    lp_count['global'] += 1
    lp_count[typ] += 1
    return "%03d/%03d %s %.3g:%.3g MiB" % (
        lp_count['global'], lp_count[typ],
        time.asctime(),
        traffic['send']/1048576.0, traffic['recv']/1048576.0)

target_name = sys.argv[1]
input_file_name = sys.argv[2]
output_file_name = sys.argv[3]
script_name = sys.argv[4]
script_schema = "TEST_SCHEMA"
current_user = "sys"
current_schema = "TEST_SCHEMA"
scope_user = "sys"
script_code = open(script_name).read()
connection_id = 666
database_name = "TESTPROC"
database_version = "TESTVER"
node_count = 1
node_id = 0
vm_id = 123
session_id = 1234
statement_id = 12
memory_limit = 2147483648
pong_time = time.time()

socket_connect = False
socket_name = None
redirector_name = None
redir_thread = None

if target_name.startswith("unix:"):
    socket_name = "ipc://" + target_name[len("unix:"):]
elif target_name.startswith("redir:"):
    redirector_name = "tcp://" + target_name[len("redir:"):]
    socket_connect = True
elif target_name.startswith("tcp:"):
    socket_name = "tcp://" + target_name[len("tcp:"):]
    socket_connect = True
else: raise SystemError("Unknown target type: %s\n" + repr(target_name))

initialized = False
closed = False
finished = False
request = exascript_request()
response = exascript_response()
meta_info = ''

def parse_coltype(coltype):
    coltype = coltype.upper()
    if coltype == 'STRING': return (PB_STRING, lambda x: x.decode('utf-8'))
    elif coltype == 'INT32': return (PB_INT32, lambda x: int(x))
    elif coltype == 'INT64': return (PB_INT64, lambda x: int(x))
    elif coltype == 'NUMERIC': return (PB_NUMERIC, lambda x: str(x))
    elif coltype == 'DOUBLE': return (PB_DOUBLE, lambda x: float(x))
    elif coltype == 'DATE': return (PB_DATE, lambda x: str(x))
    elif coltype == 'TIMESTAMP': return (PB_TIMESTAMP, lambda x: str(x))
    else: raise RuntimeError("Unknwon column type: " + repr(coltype))

def parse_number(num):
    if num.upper() == 'NONE': return None
    return int(num)

input_type = 'SCALAR'
output_type = 'RETURNS'
input_columns = []
output_columns = []
input_file = csv.reader(open(input_file_name), skipinitialspace = True)
output_file_fd = open(output_file_name, 'w')
output_file = csv.writer(output_file_fd)
script_lines = [x.strip().lower() for x in script_code.split('\n')]
script_input_defs = []
script_output_defs = []
if script_name.endswith('.java'): comment = '//'
else: comment = '#'
for line in script_lines:
    if line.startswith(comment + 'input_column:'):
        script_input_defs.append(line[len(comment + 'input_column:'):])
    elif line.startswith(comment + 'output_column:'):
        script_output_defs.append(line[len(comment + 'output_column:'):])
    elif line.startswith(comment + 'input_type:'):
        input_type = line[len(comment + 'input_type:'):].strip().upper()
    elif line.startswith(comment + 'output_type:'):
        output_type = line[len(comment + 'output_type:'):].strip().upper()
input_columns_count = len(script_input_defs)
output_columns_count = len(script_output_defs)

if input_type not in ('SCALAR', 'SET'):
    raise SystemError("Input type sholud be 'SCALAR' or 'SET', but given value is: " + repr(input_type))
if output_type not in ('RETURNS', 'EMITS'):
    raise SystemError("Output type sholud be 'RETURNS' or 'EMITS', but given value is: " + repr(output_type))

for xic, cols, defs in [(input_columns_count, input_columns, script_input_defs),
                        (output_columns_count, output_columns, script_output_defs)]:
    col_defs = csv.reader(defs, skipinitialspace = True)
    for col in range(xic):
        colname, coltype, coltypename, colsize, colprec, colscale = col_defs.next()
        coltype, colread = parse_coltype(coltype)
        colsize = parse_number(colsize)
        colprec = parse_number(colprec)
        colscale = parse_number(colscale)
        cols.append((col, colname, coltype, colread, coltypename, colsize, colprec, colscale))

next_functions = []
def generate_next(colid, colread, valattr, msgsizefun):
    def next_fun(row, values):
        val = row[colid]
        if val == '': response.next.table.data_nulls.append(True)
        else:
            getattr(values, valattr).append(colread(val))
            response.next.table.data_nulls.append(False)
        response.next.table.row_number.append(0)
        return msgsizefun(val)
    next_functions.append(next_fun)
for colid, colname, coltype, colread, coltypename, colsize, colprec, colscale in input_columns:
    if coltype == PB_DOUBLE: generate_next(colid, colread, 'data_double', lambda v: 12)
    elif coltype == PB_INT32: generate_next(colid, colread, 'data_int32', lambda v: 4)
    elif coltype == PB_INT64: generate_next(colid, colread, 'data_int64', lambda v: 4)
    elif coltype == PB_BOOLEAN: generate_next(colid, colread, 'data_bool', lambda v: 1)
    elif coltype in (PB_NUMERIC, PB_TIMESTAMP, PB_DATE, PB_STRING):
        generate_next(colid, colread, 'data_string', lambda v: len(v))
    else: raise RuntimeError("Unuspported column type")

emit_functions = []
def generate_emit(typ, enc):
    def emit_enc(cur, row):
        cn, ct = cur['null'], cur[typ]
        if request.emit.table.data_nulls[cn[0]] == True:
            row.append('')
        else:
            row.append(unicode(ct[1][ct[0]]).encode('utf-8'))
            ct[0] += 1
        cn[0] += 1
    def emit_raw(cur, row):
        cn, ct = cur['null'], cur[typ]
        if request.emit.table.data_nulls[cn[0]] == True:
            row.append('')
        else:
            row.append(ct[1][ct[0]])
            ct[0] += 1
        cn[0] += 1
    if enc: emit_functions.append(emit_enc)
    else: emit_functions.append(emit_raw)
for colid, colname, coltype, colread, coltypename, colsize, colprec, colscale in output_columns:
    if coltype == PB_DOUBLE: generate_emit('double', False)
    elif coltype == PB_INT32: generate_emit('int32', False)
    elif coltype == PB_INT64: generate_emit('int64', False)
    elif coltype == PB_BOOLEAN: generate_emit('bool', False)
    elif coltype in (PB_NUMERIC, PB_TIMESTAMP, PB_DATE, PB_STRING):
        generate_emit('string', True)

class ping(threading.Thread):
    def run(self):
        global meta_info
        req = exascript_request()
        rep = exascript_response()
        req.type = MT_PING_PONG
        req.connection_id = connection_id
        req.ping.meta_info = meta_info
        pingreq = req.SerializeToString()
        while not closed:
            zredir.send(pingreq)
            rep.Clear()
            rep.ParseFromString(zredir.recv())
            if rep.connection_id != connection_id:
                raise RuntimeError("Wrong connection id in ping response")
            if rep.type == MT_CLOSE:
                raise RuntimeError("CLOSE message from redirector: %s" % repr(rep.close.exception_message))
            if rep.type != MT_PING_PONG:
                raise RuntimeError("Wrong ping response from redirector")
            meta_info = rep.ping.meta_info
            print "PONG:", time.asctime()
            if not closed: time.sleep(5)

zcontext = zmq.Context()
if redirector_name != None:
    zredir = zcontext.socket(zmq.REQ)
    zredir.connect(redirector_name)
    request.Clear()
    request.type = MT_INFO
    request.connection_id = connection_id
    request.info.database_name = database_name
    request.info.database_version = database_version
    request.info.source_code = script_code
    request.info.script_name = script_name
    request.info.current_user = current_user
    request.info.current_schema = current_schema
    request.info.scope_user = scope_user
    request.info.script_schema = script_schema
    request.info.session_id = session_id
    request.info.statement_id = statement_id
    request.info.node_count = node_count
    request.info.node_id = node_id
    request.info.vm_id = vm_id
    request.info.maximal_memory_limit = memory_limit
    #request.info.vm_type = PB_VM_EXTERNAL
    request.info.meta_info = meta_info
    while True:
        zredir.send(request.SerializeToString())
        response.Clear()
        response.ParseFromString(zredir.recv())
        if response.type == MT_TRY_AGAIN:
            time.sleep(1)
            continue
        elif response.type == MT_CLOSE:
            raise RuntimeError("Error: %s" % repr(response.close.exception_message))
        elif response.type == MT_CLIENT:
            socket_name = response.client.client_name
            meta_info = response.client.meta_info
            break
        else: raise RuntimeError("Unknown redirector response")
    redir_thread = ping()
    redir_thread.start()

zsocket = zcontext.socket(zmq.REP)
if socket_connect:
    print "Connect to VM with socket:", socket_name
    zsocket.connect(socket_name)
else:
    print "Wait for VM connection on socket:", socket_name
    zsocket.bind(socket_name)

def send_message(msg):
    traffic['send'] += len(msg)
    return zsocket.send(msg)

def handle_client():
    global initialized
    response.Clear()
    response.type = MT_INFO
    response.connection_id = connection_id
    response.info.database_name = database_name
    response.info.database_version = database_version
    response.info.source_code = script_code
    response.info.script_name = script_name
    response.info.script_schema = script_schema
    response.info.current_user = current_user
    response.info.current_schema = current_schema
    response.info.scope_user = scope_user
    response.info.session_id = session_id
    response.info.statement_id = statement_id
    response.info.node_count = node_count
    response.info.node_id = node_id
    response.info.vm_id = vm_id
    #response.info.vm_type = PB_VM_EXTERNAL
    response.info.maximal_memory_limit = memory_limit
    response.info.meta_info = meta_info
    #if redirector_name != None: response.info.vm_type = PB_VM_EXTERNAL
    #else:
    #    if script_name.endswith('.py'): response.info.vm_type = PB_VM_PYTHON
    #    elif script_name.endswith('.R'): response.info.vm_type = PB_VM_R
    #    elif script_name.endswith('.java'): response.info.vm_type = PB_VM_JAVA
    #    else: raise RuntimeError("Unsupported VM type")
    send_message(response.SerializeToString())
    initialized = True

def handle_meta():
    response.Clear()
    response.type = MT_META
    response.connection_id = connection_id
    response.meta.single_call_mode = False

    if input_type == 'SCALAR': response.meta.input_iter_type = PB_EXACTLY_ONCE
    else: response.meta.input_iter_type = PB_MULTIPLE
    if output_type == 'RETURNS': response.meta.output_iter_type = PB_EXACTLY_ONCE
    else: response.meta.output_iter_type = PB_MULTIPLE
    
    for xic, xadd in [(input_columns, response.meta.input_columns),
                      (output_columns, response.meta.output_columns)]:
        for colid, colname, coltype, colread, coltypename, colsize, colprec, colscale in xic:
            col = xadd.add()
            col.name = colname
            col.type = coltype
            col.type_name = coltypename
            if colsize != None: col.size = colsize
            if colprec != None: col.precision = colprec
            if colscale != None: col.scale = colscale
    send_message(response.SerializeToString())

def handle_close():
    if request.close.exception_message != None:
        raise RuntimeError(request.close.exception_message)
    response.Clear()
    response.type=MT_FINISHED
    response.connection_id = connection_id
    send_message(response.SerializeToString())
    global closed
    closed = True
    print lp('global'), 'close everything'
    output_file_fd.close()


def handle_import():
    response.Clear()
    response.type = MT_IMPORT
    response.connection_id = connection_id
    fname = getattr(request, 'import').script_name
    kind = getattr(request, 'import').kind
    if kind == PB_IMPORT_SCRIPT_CODE:
        if script_name.endswith('.py'): fname = fname + '.py'
        elif script_name.endswith('.R'): fname = fname + '.R'
        else: fname = fname + '.java'
        try: getattr(response, 'import').source_code = open(fname).read()
        except Exception, err:
            response.exception_message = str(err)
        send_message(response.SerializeToString())
    elif kind == PB_IMPORT_CONNECTION_INFORMATION:
        try:
            getattr(response, 'import').connection_information.kind = 'connection'
            getattr(response, 'import').connection_information.address = 'some_sql_database'  
            getattr(response, 'import').connection_information.user = 'some_user_you_cannot_guess!!'
            getattr(response, 'import').connection_information.password = 'some_password_you_cannot_guess!!'
        except Exception, err:
            response.exception = str(err)
        send_message(response.SerializeToString())

nextpacket = None; restrows = []; emited_rows = sys.maxint; lastpacket = False
def generate_nextpacket():
    global nextpacket, restrows, lastpacket
    response.Clear(); response.connection_id = connection_id
    message_size = 0; thisrows = []
    if emited_rows < sys.maxint:
        restpos = 0; restlimit = min(len(restrows), emited_rows)
        while message_size <= SWIG_MAX_VAR_DATASIZE and restpos < restlimit:
            row = restrows[restpos]
            for col in input_columns:
                message_size += next_functions[col[0]](row, response.next.table)
            restpos += 1
        thisrows, restrows = restrows[:restpos], restrows[restpos:]
    if not lastpacket:
        while message_size <= SWIG_MAX_VAR_DATASIZE and emited_rows > len(thisrows):
            try: row = input_file.next()
            except StopIteration, err: break
            thisrows.append(row)
            for col in input_columns:
                message_size += next_functions[col[0]](row, response.next.table)
    if len(thisrows) == 0:
        response.type = MT_DONE
        lastpacket = True
    elif request.type == MT_RESET: response.type = MT_RESET
    else: response.type = MT_NEXT
    response.next.table.rows = len(thisrows)
    response.next.table.rows_in_group = 0
    nextpacket = (len(thisrows), response.SerializeToString(), thisrows)

sendpacket = None
def handle_next():
    global nextpacket, restrows, sendpacket, finished
    if sendpacket != None and sendpacket[0] > emited_rows:
        restrows = sendpacket[2][emited_rows:] + nextpacket[2] + restrows
        nextpacket = None
    if nextpacket == None:
        generate_nextpacket()
    sendpacket = nextpacket; nextpacket = None
    send_message(sendpacket[1])
    if emited_rows != sys.maxint:
        print lp('next'), "sent", sendpacket[0], "rows"
    else: print lp('next'), "sent", sendpacket[0], "rows"
    if lastpacket and len(restrows) == 0:
        finished = True
    else: generate_nextpacket()        

def handle_emit():
    global emited_rows
    response.Clear()
    response.type = MT_EMIT
    response.connection_id = connection_id
    send_message(response.SerializeToString())

    current = {}
    current['null'] = [0, None]
    current['string'] = [0, request.emit.table.data_string]
    current['bool'] = [0, request.emit.table.data_bool]
    current['int32'] = [0, request.emit.table.data_int32]
    current['int64'] = [0, request.emit.table.data_int64]
    current['double'] = [0, request.emit.table.data_double]

    rows = []
    for rownum in range(request.emit.table.rows):
        row = []
        for col in output_columns:
            emit_functions[col[0]](current, row)
        rows.append(row)
    output_file.writerows(rows)
    if input_type == 'SCALAR' and output_type == 'RETURNS' and request.emit.table.rows > 0:
        emited_rows = request.emit.table.rows
    print lp('emit'), "emit", request.emit.table.rows, "rows"

def handle_run():
    response.Clear()
    if finished: response.type = MT_CLEANUP
    else: response.type = MT_RUN
    response.connection_id = connection_id
    send_message(response.SerializeToString())

def handle_done():
    response.Clear()
    if finished: response.type = MT_CLEANUP
    else: response.type = MT_DONE
    response.connection_id = connection_id
    send_message(response.SerializeToString())

def handle_finished():
    global closed
    response.Clear()
    response.type = MT_FINISHED
    response.connection_id = connection_id
    send_message(response.SerializeToString())
    closed = True

def main():
    global closed
    try:
        while not closed:
            msg = zsocket.recv()
            traffic['recv'] += len(msg)
            request.ParseFromString(msg)
    
            if request.type == MT_CLIENT:
                handle_client()
                continue
            elif not initialized:
                raise RuntimeError("Wrong request, first request should be MT_CLIENT")
            elif request.connection_id != connection_id:
                raise RuntimeError("Wrong connection ID")
    
            if request.type == MT_META: handle_meta()
            elif request.type == MT_CLOSE: handle_close()
            elif request.type == MT_IMPORT: handle_import()
            elif request.type == MT_NEXT: handle_next()
            elif request.type == MT_RESET: handle_next()
            elif request.type == MT_EMIT: handle_emit()
            elif request.type == MT_RUN: handle_run()
            elif request.type == MT_DONE: handle_done()
            elif request.type == MT_FINISHED: handle_finished()
            else: raise RuntimeError("Unknown request type")
    finally:
        closed = True
      
#import trace  
#tracer = trace.Trace(ignoredirs = [sys.prefix, sys.exec_prefix], trace = 1)
#tracer.run("main()")
#tracer.results().write_results(show_missing = True)
main()
