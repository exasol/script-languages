import sys
import pyodbc

class ClientError(Exception):
    pass

class ODBCClient(object):
    def __init__(self, dsn):
        self.dsn = dsn
        self.cursor = None
        self.params = {}
        self.params['dsn'] = dsn
        self.params['uid'] = 'sys'
        self.params['pwd'] = 'exasol'

    def connect(self, **kwargs):
        params = self.params.copy()
        params.update(kwargs)
        self.conn = pyodbc.connect(**params)
        self.conn.setencoding(str, encoding='utf-8') 
        self.conn.setencoding(unicode, encoding='utf-8') 
        self.cursor = self.conn.cursor()
        self._setScriptLanguagesFromArgs()

    def _setScriptLanguagesFromArgs(self):
        for i, arg in enumerate(sys.argv):
            if arg == '--script-languages':
                if len(sys.argv) == i + 1:
                    raise ClientError('Value for --script-languages missing')
                self.query("ALTER SESSION SET SCRIPT_LANGUAGES='%s'" % sys.argv[i + 1])
                break

    def query(self, qtext, *args):
        if self.cursor is None:
            raise ClientError('query() requires connect() first')
        q = self.cursor.execute(qtext, *args)
        try:
            return q.fetchall()
        except pyodbc.ProgrammingError as e:
            if 'No results.  Previous SQL was not a query.' in str(e):
                return None
            else:
                raise

    def executeStatement(self, qtext, *args):
        if self.cursor is None:
            raise ClientError('executeStatement() requires connect() first')
        self.cursor.execute(qtext, *args)
        return self.cursor.rowcount

    def columns(self, table=None, catalog=None, schema=None, column=None):
        args = {}
        if table:
            args['table'] = table
        if catalog:
            args['catalog'] = catalog
        if schema:
            args['schema'] = schema
        if column:
            args['column'] = column
        return self.cursor.columns(**args).fetchall()

    def rowcount(self):
        return self.cursor.rowcount
        
    def cursorDescription(self):
        return self.cursor.description

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def close(self):
        self.cursor.close()
        self.conn.close()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
