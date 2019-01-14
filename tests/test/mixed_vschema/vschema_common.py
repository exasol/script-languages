
import sys
import os
sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
import exatest
import udf
#import concurrent.futures
from multiprocessing import Manager

from exatest.clients.odbc import ODBCClient

# FEEDBACK:
# setUpClass is a classmethod and thus doesn't have query ability. In unicode.py, exaplus is called manually
# added assertRowsEqualIgnoreOrder
# new methods (compareColumn, columnNames/description, setColumn...)
# Added queryScalar convenience method
# Exception is shown in log, although we use assertRaisesRegexp (does not happen if I create a new ODBCClient connection and exception happens there
# Not sure if Python Tests are executed unoptimized? Should be possible, to test assertions too.
# Offer separate Connection class. To separate concerns, and so that all things are available in additional connection
# Write Testing Guide (what you need to know when writing Python ExaTests)
# - 1 > None => If you check for things like >, or <, you should check for None extra, or test for = NULL in sql
# - Expected value first or last (as in unittest documentation)
# Offer self.queryUnicode, which automatically converts str fields in resultset to unicode.
# optionally result as pandas dataframe (contains column names, etc.)?

class VSchemaTest(exatest.TestCase):

    # TODO Maybe move this to testcase.py
    def queryScalarUnicode(self, *args, **kwargs):
        ''' Runs a scalar query, and converts the resulting string to unicode (if it is not already) '''
        stringVal = self.queryScalar(*args, **kwargs)
        # Only makes sense if this is a string
        self.assertTrue(isinstance(stringVal, basestring))
        if isinstance(stringVal, str):
            return stringVal.decode('utf-8')
        else:
            return stringVal

    def queryUnicode(self, *args, **kwargs):
        ''' Runs a scalar query, and converts all fields of type "str" to unicode, assuming utf-8 encoding '''
        rows = self.query(*args, **kwargs)
        return self.decodeUtf8Fields(rows)

    def decodeUtf8Fields(self, rows):
        ''' Replace all string-typed fields (python type "str") in an resultset by unicode strings (python type "unicode").
            Assumes that string is encoded in utf-8. '''
        for r, row in enumerate(rows):
            for c, column in enumerate(rows[r]):
                if isinstance(column, str):
                    rows[r][c] = column.decode('utf-8')
        return rows

    def getColumn(self, rows, columnIdx):
        ''' Return a single column of a resultset as a list '''
        return [row[columnIdx] for row in rows]

    def getColumnByName(self, rows, columnName):
        ''' Return a single column of a resultset as a list '''
        return [getattr(row,columnName) for row in rows]

    def assertBetween(self, value, lower, upper):
        ''' Checks lower < value < upper '''
        self.assertGreater(value, lower)
        self.assertLess(value, upper)

    def assertColumnEqualConst(self, rows, columnIdx, constValue):
        self.assertEqual([constValue]*len(rows), self.getColumn(rows,columnIdx))
    
    #def initPool(self):
    #    self.executor = concurrent.futures.ProcessPoolExecutor()
    #    self.manager = Manager()
    #    self.odbcDictionary = self.manager.dict()


    def createFastAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT {schema}.{adapter} AS
            import json
            import string
            def adapter_call(request):
                # database expects utf-8 encoded string of type str. unicode not yet supported
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    res = {{
                        "type": "createVirtualSchema",
                        "schemaMetadata": {{
                            "tables": [
                            {{
                                "name": "DUMMY",
                                "columns": [{{
                                    "name": "KEY",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }},{{
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).encode('utf-8')
                elif root["type"] == "dropVirtualSchema":
                    return json.dumps({{"type": "dropVirtualSchema"}}).encode('utf-8')
                elif root["type"] == "setProperties":
                    return json.dumps({{"type": "setProperties"}}).encode('utf-8')
                elif root["type"] == "refresh":
                    return json.dumps({{"type": "refresh"}}).encode('utf-8')
                if root["type"] == "getCapabilities":
                    return json.dumps({{
                        "type": "getCapabilities",
                        "capabilities": []
                        }}).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
                elif root["type"] == "pushdown":
                    res = {{
                        "type": "pushdown",
                        "sql": "SELECT * FROM (VALUES ('FOO', 'BAR')) t"
                    }}
                    return json.dumps(res).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

    def createTestPropertyAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT {schema}.{adapter} AS
            import json
            import string
            def adapter_call(request):
                # database expects utf-8 encoded string of type str. unicode not yet supported
                root = json.loads(request)
                if root["type"] == "createVirtualSchema":
                    res = {{
                        "type": "createVirtualSchema",
                        "schemaMetadata": {{
                            "tables": [
                            {{
                                "name": "DUMMY",
                                "columns": [{{
                                    "name": "KEY",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }},{{
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 2000000}}
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).encode('utf-8')
                elif root["type"] == "dropVirtualSchema":
                    return json.dumps({{"type": "dropVirtualSchema"}}).encode('utf-8')
                elif root["type"] == "setProperties":
                    for itemKey, itemValue in root["schemaMetadataInfo"]["properties"].iteritems():
                        for newItemKey, newItemValue in root["properties"].iteritems():
                            if (itemKey == newItemKey and itemValue == newItemValue):
                                raise ValueError('Expected different values for old (name: ' + itemKey + ' value: ' + itemValue + ') and new property (name: ' + newItemKey + ' value: ' + newItemValue +').')
                elif root["type"] == "refresh":
                    return json.dumps({{"type": "refresh"}}).encode('utf-8')
                if root["type"] == "getCapabilities":
                    return json.dumps({{
                        "type": "getCapabilities",
                        "capabilities": []
                        }}).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
                elif root["type"] == "pushdown":
                    res = {{
                        "type": "pushdown",
                        "sql": "SELECT * FROM (VALUES ('FOO', 'BAR')) t"
                    }}
                    return json.dumps(res).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

    def createFailingAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT {schema}.{adapter} AS
            import json
            import string
            def adapter_call(request):
                raise ValueError('This should never be called')
            /
            ''').format(schema = schemaName, adapter = adapterName))

    def createJdbcAdapter(self, schemaName="ADAPTER", adapterName="JDBC_ADAPTER"):
        self.dropOldAdapter(schemaName, adapterName)
        self.query('CREATE SCHEMA {schema}'.format(schema=schemaName))
        self.query(udf.fixindent('''
            CREATE OR REPLACE JAVA ADAPTER SCRIPT {adapter} AS
            %scriptclass com.exasol.adapter.jdbc.JdbcAdapter;
            %jvmoption -Xms64m -Xmx64m;
            %jar {jdbc_path};
            %jar /buckets/bfsdefault/jdbc-adapter/virtualschema-jdbc-adapter.jar;
            /
            ''').format(adapter = adapterName, jdbc_path = '/buckets/bfsdefault/jdbc-adapter/exajdbc.jar')) #udf.opts.jdbc_path)) # /buckets/internal/jdbc/EXASOL/exajdbc.jar

    def dropOldAdapter(self, schemaName, adapterName):
        vSchemasFromAdapter = self.query("SELECT SCHEMA_NAME FROM EXA_VIRTUAL_SCHEMAS WHERE ADAPTER_SCRIPT LIKE '{schema}.%'".format(schema = schemaName))
        for schema in vSchemasFromAdapter:
            self.query('DROP FORCE VIRTUAL SCHEMA IF EXISTS {schema} CASCADE'.format(schema=schema[0]))
        self.query('DROP SCHEMA IF EXISTS {schema} CASCADE'.format(schema=schemaName))

    def createVirtualSchemaJdbc(self, schemaName, remoteSchemaName, adapterName, isLocal, useConnection=False):
        self.query('DROP FORCE VIRTUAL SCHEMA IF EXISTS {schema} CASCADE'.format(schema = schemaName))
        # if useConnection:
        #     self.query("CREATE CONNECTION TEST_CONN TO 'jdbc:exa:{host_port}' USER 'sys' IDENTIFIED BY 'exasol'".format(host_port = udf.opts.server))
        #     credentialProperties = "CONNECTION_NAME='TEST_CONN'"
        # else:
        #     credentialProperties = "CONNECTION_STRING='jdbc:exa:{host_port}' USERNAME='sys' PASSWORD='exasol'".format(host_port = udf.opts.server)

        if useConnection:
            self.query("CREATE CONNECTION TEST_CONN TO 'jdbc:exa:localhost:8888' USER 'sys' IDENTIFIED BY 'exasol'")
            credentialProperties = "CONNECTION_NAME='TEST_CONN'"
        else:
            credentialProperties = "CONNECTION_STRING='jdbc:exa:localhost:8888' USERNAME='sys' PASSWORD='exasol'"
        # self.query('''
        #     CREATE VIRTUAL SCHEMA {schema} USING {adapter} WITH
        #     {credentialProperties} SCHEMA_NAME='{remoteSchema}' IS_LOCAL='{islocal}' SQL_DIALECT='EXASOL' --  DEBUG_SERVER='192.168.5.59' DEBUG_PORT='3000'
        #     EXCEPTION_HANDLING='NONE'
        #     '''.format(credentialProperties = credentialProperties, schema = schemaName, remoteSchema = remoteSchemaName, adapter = adapterName, islocal = str(isLocal)))
        # rows = self.query('''
        #     SELECT CURRENT_SCHEMA
        #     ''')
        self.query('''
            CREATE VIRTUAL SCHEMA {schema} USING {adapter} WITH
            {credentialProperties} SCHEMA_NAME='{remoteSchema}' IS_LOCAL='{islocal}' SQL_DIALECT='EXASOL'
            EXCEPTION_HANDLING='NONE'
            '''.format(credentialProperties = credentialProperties, schema = schemaName, remoteSchema = remoteSchemaName, adapter = adapterName, islocal = str(isLocal)))
        rows = self.query('''
            SELECT CURRENT_SCHEMA
            ''')

        self.assertRowEqual((schemaName,), rows[0])


    def createNative(self):
        self.query('DROP SCHEMA IF EXISTS NATIVE CASCADE')
        self.query('CREATE SCHEMA NATIVE')
        self.query('CREATE TABLE t(a int, b varchar(100), c double)')
        self.query('''
            INSERT INTO t VALUES
            (1, 'a', 1.1),
            (2, 'b', 2.2),
            (3, 'c', 3.3),
            (1, 'd', 4.4),
            (2, 'e', 5.5),
            (3, 'f', 6.6),
            (null, null, null)
            ''')
        self.query('CREATE TABLE g (k int, v1 int, v2 varchar(100))')
        self.query('''
            INSERT INTO g VALUES
            (1, 1, 'one'),
            (1, 1, 'two'),
            (1, 1, 'three'),
            (1, 2, 'one'),
            (1, 2, 'two'),
            (1, 2, 'three'),
            (2, 1, 'one'),
            (2, 2, 'one'),
            (2, 3, 'one'),
            (2, 3, 'two'),
            (3, 1, 'three'),
            (3, 1, 'three'),
            (3, 2, 'two'),
            (3, 3, 'one')
            ''')
        self.query(''' CREATE TABLE t_datetime(a timestamp, b date) ''')
        self.query('''
            INSERT INTO t_datetime VALUES
            (DATE '2015-12-01 12:30:01.1234', '2015-12-01'),
            (DATE '2014-12-01 12:30:01.1234', '2014-12-01');
            ''')
        self.query(''' CREATE TABLE test(a timestamp with local time zone) ''')
        self.query('''
            INSERT INTO test VALUES
            (DATE '2015-12-01 12:30:01.1234');
            ''')

        #TODO: Support interval types
        self.query(''' CREATE TABLE t_interval(a interval year to month, b interval day to second) ''')
        self.query('''
            INSERT INTO t_interval VALUES
            (INTERVAL '5' MONTH, INTERVAL '5' DAY);''')
        self.query('''
            INSERT INTO t_interval VALUES
            (INTERVAL '130' MONTH (3), INTERVAL '100' HOUR(3));''')
        self.query('''
            INSERT INTO t_interval VALUES
            (INTERVAL '27' YEAR, INTERVAL '6' MINUTE);''')
        self.query('''
            INSERT INTO t_interval VALUES
            (INTERVAL '2-1' YEAR TO MONTH, INTERVAL '1.99999' SECOND(2,2));''')
        self.query('''
            INSERT INTO t_interval VALUES
            (INTERVAL '5' MONTH, INTERVAL '2 23:10:59' DAY TO SECOND);''')
        self.query('''
            INSERT INTO t_interval VALUES
            (INTERVAL '5' MONTH, INTERVAL '23:10:59.123' HOUR(2) TO SECOND(3));''')

        self.query(''' CREATE TABLE t_geometry(id int, a geometry) ''')
        self.query('''
            INSERT INTO t_geometry VALUES
            (1, 'POLYGON((1 1, 2 3, 3 4, 5 2, 4 1, 1 1) , (2 2, 3 3, 3 2, 2 2))'),
            (2, 'LINESTRING (0 0, 0 1, 1 1)'),
            (3, 'GEOMETRYCOLLECTION(POINT(2 5), LINESTRING(1 1, 15 2, 15 10))');
            ''')

        self.query(''' CREATE TABLE t_connect(val int, parent int) ''')
        self.query('''
            INSERT INTO t_connect VALUES
            (100, NULL),
            (20, 100),
            (32, 100),
            (16, 20),
            (5, 20),
            (1, 16),
            (8, 16),
            (23, 32),
            (0, 32)
            ;
            ''')
        self.query('CREATE TABLE t_nulls(a int, b varchar(100))')
        self.query('''
            INSERT INTO t_nulls VALUES
            (1, 'a'),
            (2, null),
            (3, 'c'),
            (1, null),
            (2, 'e'),
            (3, 'f'),
            (null, 'g')
            ''')
        self.query('''create table numbers1(a int, b int, c int, d int);''')
        self.query('''insert into numbers1 VALUES (1,1,1,1), (2,2,2,2), (3,3,3,3), (4,4,4,4), (5,5,5,5), (6,6,6,6);''')
        self.query('''insert into numbers1 VALUES (1,2,3,4), (2,3,4,5), (3,4,5,6), (4,5,6,7), (5,6,7,8), (6,7,8,9);''')
        self.query('''insert into numbers1 VALUES (1,3,4,5), (2,4,5,6), (3,5,6,7), (4,6,7,8), (5,7,8,9), (6,8,9,10);''')
        self.query('''insert into numbers1 VALUES (1,3,5,7), (2,4,6,8), (3,5,7,9), (4,6,8,10), (5,7,9,11), (6,8,10,12);''')
        self.query('''create table numbers2(e int, f int, g int, h int);''')
        self.query('''insert into numbers2 VALUES (1,1,1,1), (2,2,2,2), (3,3,3,3), (4,4,4,4), (5,5,5,5), (6,6,6,6);''')
        self.query('''insert into numbers2 VALUES (4,3,2,1), (5,4,3,2), (6,5,4,3), (7,6,5,4), (8,7,6,5), (9,8,7,6);''')
        self.query('''insert into numbers2 VALUES (7,5,3,1), (8,6,4,2), (9,7,5,3), (10,8,6,4), (11,9,7,5), (12,10,8,6);''')

        self.query('''CREATE TABLE t_datatypes(a1 int, a2 double, a3 date,
        a4 timestamp, a5 varchar(3000), a6 char(10), a7 bool, a8 interval day to second, a9 interval year to month, a10 geometry,
        a11 decimal(10,5), a12 double precision, a13 bigint, a14 decimal, a15 decimal(29), a16 dec, a17 dec(25),
        a18 dec(27,9), a19 float, a20 integer, a21 number, a22 number(1), a23 number(3,2), a24 numeric,
        a25 numeric(6), a26 numeric(6,3), a27 real, a28 shortint, a29 smallint, a30 tinyint, a31 date, a32 timestamp with local time zone);''')

    def queryCurrentTimestamp(self):
        return self.queryScalar('SELECT CURRENT_TIMESTAMP')

    def getLastSchemaRefresh(self, schema):
        return self.queryScalar('''
            SELECT
              LAST_REFRESH
            FROM EXA_VIRTUAL_SCHEMAS
            WHERE SCHEMA_NAME = '{schema}'
            '''.format(schema=schema))

    def getLastTableRefresh(self, schema, table):
        return self.queryScalar('''
            SELECT
              LAST_REFRESH
            FROM EXA_DBA_VIRTUAL_TABLES
            WHERE TABLE_SCHEMA = '{schema}' AND TABLE_NAME = '{table}'
            '''.format(schema=schema, table=table))

    def queryColumnMetadata(self, schema):
        return self.query(u'''
            SELECT
              t.TABLE_NAME TABLE_NAME,
              c.COLUMN_NAME COLUMN_NAME,
              c.COLUMN_TYPE COLUMN_TYPE
            FROM EXA_ALL_TABLES t
              JOIN EXA_DBA_VIRTUAL_TABLES vt ON t.TABLE_SCHEMA = vt.TABLE_SCHEMA AND t.TABLE_NAME = vt.TABLE_NAME AND t.TABLE_OBJECT_ID = vt.TABLE_OBJECT_ID
              JOIN EXA_DBA_COLUMNS c ON t.TABLE_SCHEMA = c.COLUMN_SCHEMA AND t.TABLE_NAME = c.COLUMN_TABLE
              JOIN EXA_DBA_VIRTUAL_COLUMNS vc ON c.COLUMN_SCHEMA = vc.COLUMN_SCHEMA AND c.COLUMN_TABLE = vc.COLUMN_TABLE AND c.COLUMN_NAME = vc.COLUMN_NAME and c.COLUMN_OBJECT_ID = vc.COLUMN_OBJECT_ID
            WHERE t.TABLE_SCHEMA = '{schema}'
            ORDER BY t.TABLE_NAME, c.COLUMN_NAME
            '''.format(schema=schema))

    def compareWithProfilingExtended(self, query, ignoreOrder=False, partialOrder=-1, profilingResponse=None):
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        replace_dict_native = {"v": "NATIVE", "v2": "NATIVE", "v3": "NATIVE", "n": "NATIVE"}
        self.assertGreater(query.find("v"), -1, msg="Query has no placeholder for schema: " + query)
        with self.expectations():
            self.query("ALTER SESSION SET PROFILE='on'")
            self.commit()
            self.compareWithNativeExpected(query, replace_dict_virtual, replace_dict_native, ignoreOrder, partialOrder)
            self.query("ALTER SESSION SET PROFILE='off'")
            self.commit()
            self.query("FLUSH STATISTICS")
            self.commit()
            replacedQuery = query.format(**replace_dict_virtual)
            profileQuery = '''SELECT REMARKS FROM EXA_DBA_PROFILE_LAST_DAY WHERE SESSION_ID =
            CURRENT_SESSION and STMT_ID > CURRENT_STATEMENT - 7 and PART_NAME='PUSHDOWN' and SQL_TEXT='{query}' ORDER BY PART_ID'''.format(query=replacedQuery)
            rows = self.query(profileQuery)
            self.expectRowsEqual(rows, profilingResponse, msg="assertRowsEqual failed for query '" + profileQuery + "'\nActual: " + str(rows) + "\nExpected" + str(profilingResponse))
        self.assertExpectations()



    def compareWithNativeExtended(self, query, ignoreOrder=False, partialOrder=-1, explainResponse=None):
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        replace_dict_native = {"v": "NATIVE", "v2": "NATIVE", "v3": "NATIVE", "n": "NATIVE"}
        self.assertGreater(query.find("v"), -1, msg="Query has no placeholder for schema: " + query)
        with self.expectations():
          if explainResponse != None:
              explainQuery = "EXPLAIN VIRTUAL " + query.format(**replace_dict_virtual)
              rows = self.query(explainQuery)
              if (isinstance(explainResponse, basestring)):
                  self.expectTrue(explainResponse in rows[0][1], "Expected string " + explainResponse + " not found in explain virtual query '" + explainQuery + "'\nActual: " + str(rows) + "\nExpected: " + explainResponse)
              else:
                  i = 0;
                  lrows = [x for x in explainResponse]
                  rrows = [x[1] for x in rows]
                  if lrows[0].find("IMPORT INTO") == -1:
                      rrows = [x.split("STATEMENT ")[1] if x.find("IMPORT INTO") != -1 else x for x in rrows]

                  lrows = sorted(lrows)
                  rrows = sorted(rrows)
                  for item in lrows:
                      value = rrows[i]
                      if (len(item) == len(value)):
                          self.expectEqual(item, value, msg="Expected string " + str(explainResponse) + " not found in explain virtual query '" + explainQuery + "'\nActual: " + str(value) + "\nExpected: " + str(item))
                      else :
                          self.expectTrue(item in value, "Expected string " + str(explainResponse) + " not found in explain virtual query '" + explainQuery + "'\nActual: " + str(value) + "\nExpected: " + str(item))
                      i += 1
          self.compareWithNativeExpected(query, replace_dict_virtual, replace_dict_native, ignoreOrder, partialOrder)
        self.assertExpectations()


    def compareWithNativeSimple(self, query, ignoreOrder=False, partialOrder=-1):
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        replace_dict_native = {"v": "NATIVE", "v2": "NATIVE", "v3": "NATIVE", "n": "NATIVE"}
        self.assertGreater(query.find("v"), -1, msg="Query has no placeholder for schema: " + query)
        self.compareWithNative(query, replace_dict_virtual, replace_dict_native, ignoreOrder, partialOrder)

    def compareWithNativeSimpleExpected(self, query, ignoreOrder=False, partialOrder=-1):
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        replace_dict_native = {"v": "NATIVE", "v2": "NATIVE", "v3": "NATIVE", "n": "NATIVE"}
        self.expectGreater(query.find("v"), -1, msg="Query has no placeholder for schema: " + query)
        self.compareWithNativeExpected(query, replace_dict_virtual, replace_dict_native, ignoreOrder, partialOrder)

    def compareWithNative(self, query, replace_dict_virtual, replace_dict_native, ignoreOrder, partialOrder):
        rows = self.query(query.format(**replace_dict_virtual))
        rows_native = self.query(query.format(**replace_dict_native))
        if ignoreOrder:
            self.assertRowsEqualIgnoreOrder(rows, rows_native, msg="assertRowsEqualIgnoreOrder failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))
        elif partialOrder > -1:
            self.assertRowsEqualIgnoreOrder(rows, rows_native, msg="assertRowsEqualIgnoreOrder failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))
            self.assertEqual(self.getColumn(rows, partialOrder), self.getColumn(rows_native, partialOrder), msg="assertEqual failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))
        else:
            self.assertEqual(rows, rows_native, msg="assertEqual failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))

    def compareWithNativeExpected(self, query, replace_dict_virtual, replace_dict_native, ignoreOrder, partialOrder):
        rows = self.query(query.format(**replace_dict_virtual))
        rows_native = self.query(query.format(**replace_dict_native))
        if ignoreOrder:
            self.expectRowsEqualIgnoreOrder(rows, rows_native, msg="assertRowsEqualIgnoreOrder failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))
        elif partialOrder > -1:
            self.expectRowsEqualIgnoreOrder(rows, rows_native, msg="assertRowsEqualIgnoreOrder failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))
            self.expectEqual(self.getColumn(rows, partialOrder), self.getColumn(rows_native, partialOrder), msg="assertEqual failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))
        else:
            self.expectEqual(rows, rows_native, msg="assertEqual failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))

    def compareQueriesWithNative(self, queries, ignoreOrder=False):
        # TODO Loop for all capability combinations
        # SELECTLIST_PROJECTION, SELECTLIST_EXPRESSIONS
        # FILTER_EXPRESSIONS
        # AGGREGATE_SINGLE_GROUP, AGGREGATE_GROUP_BY_COLUMN, AGGREGATE_GROUP_BY_EXPRESSION, AGGREGATE_GROUP_BY_TUPLE, AGGREGATE_HAVING
        # ORDER_BY_COLUMN, ORDER_BY_EXPRESSION
        # LIMIT, LIMIT_WITH_OFFSET
        queryList = TestUtils.queriesFromExaplusScript(queries)
        for q in queryList:
            # print("Run Query: " + q)
            self.compareWithNativeSimple(q, ignoreOrder)

    def compareWithExplainVirtual(self, query, pushdownResponse):
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        rows = self.query("EXPLAIN VIRTUAL " + query.format(**replace_dict_virtual))
        if (isinstance(pushdownResponse, basestring)):
            rows_explain = [(pushdownResponse)]
        else:
            rows_explain = []
            i = 1;
            for item in pushdownResponse:
                rows_explain.append((item))
                i += 1
        self.assertRowsEqualIgnoreOrder(self.getColumn(rows,1), rows_explain, msg="assertEqual failed for explain virtual query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_explain))

    def compareQueriesWithExplainVirtual(self, queries, pushdownResponses):
        queryList = TestUtils.queriesFromExaplusScript(queries)
        i = 0;
        for q in queryList:
            # print("Run Query: " + q)
            self.compareWithExplainVirtual(q, pushdownResponses[i])
            i += 1

    def completeFunctionTest(self, functionCall, table, explainResponse, whereClause = ''):
        query = "SELECT " + functionCall + " FROM {v}." + table + " " + whereClause;
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        explainQuery = "EXPLAIN VIRTUAL " + query.format(**replace_dict_virtual)
        rows = self.query(explainQuery)
        with self.expectations():
          self.expectTrue(explainResponse in rows[0][1], "Expected string " + explainResponse + " not found in explain virtual query '" + explainQuery + "'\nActual: " + str(rows) + "\nExpected: " + explainResponse)
          self.compareWithNativeSimple(query, True)
        self.assertExpectations()

    def completeFunctionTestPar(self, functionCall, table, explainResponse, whereClause = ''):
        client = ODBCClient('exatest')
        try:
            client.connect()
        except Exception as e:
            self.log.critical(str(e))
            raise
        query = "SELECT " + functionCall + " FROM {v}." + table + " " + whereClause;
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        explainQuery = "EXPLAIN VIRTUAL " + query.format(**replace_dict_virtual)
        rows = queryPar(self, client, explainQuery)
        self.assertTrue(explainResponse in rows[0][1], "Function call " + functionCall + " not found in explain virtual query '" + explainQuery + "'\nActual: " + str(rows) + "\nExpected" + explainResponse)
        self.compareWithNativeSimplePar(client, query, True)
        client.close()

    def compareWithNativeSimplePar(self, client, query, ignoreOrder=False):
        replace_dict_virtual = {"v": "VS1", "v2": "VS2", "v3": "VS3", "n": "NATIVE"}
        replace_dict_native = {"v": "NATIVE", "v2": "NATIVE", "v3": "NATIVE", "n": "NATIVE"}
        self.assertGreater(query.find("v"), -1, msg="Query has no placeholder for schema: " + query)
        self.compareWithNativePar(client, query, replace_dict_virtual, replace_dict_native, ignoreOrder)

    def compareWithNativePar(self, client, query, replace_dict_virtual, replace_dict_native, ignoreOrder):
        rows = queryPar(self, client, query.format(**replace_dict_virtual))
        rows_native = queryPar(self, client, query.format(**replace_dict_native))
        if ignoreOrder:
            self.assertRowsEqualIgnoreOrder(rows, rows_native, msg="assertRowsEqualIgnoreOrder failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))
        else:
            self.assertEqual(rows, rows_native, msg="assertEqual failed for query '" + query + "'\nActual: " + str(rows) + "\nExpected" + str(rows_native))

def queryPar(master, client, *args, **kwargs):
    master.log.debug('executing SQL: %s', args[0])
    try:
        return client.query(*args)
    except Exception as e:
        if not kwargs.get('ignore_errors'):
            master.log.error('executing SQL failed: %s: %s', e.__class__.__name__, e)
            if not log.isEnabledFor(logging.DEBUG):
                master.log.error('executed SQL was: %s', args[0])
            raise

class TestUtils:
    @classmethod
    def queriesFromExaplusScript(cls, queryScript):
        '''Gets a string with one or more queries, in exaplus-compatible format,
        and returns a list of strings with the individual queries.
        
        CREATE SCRIPT is not yet supported, but will be added.
        '''
        queryList = queryScript.split(';')
        queryList = map(lambda a: a.strip(), queryList)
        queryList = map(cls.removeCommentLines, queryList)
        queryList = [q for q in queryList if q != '']
        return queryList
        
    @classmethod
    def removeCommentLines(cls, query):
        queryLines = query.split("\n")
        queryLines = [q for q in queryLines if not q.lstrip().startswith("--")]
        return "\n".join(queryLines)
