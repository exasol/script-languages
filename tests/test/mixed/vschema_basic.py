#!/usr/opt/bs-python-2.7/bin/python
# -*- coding: utf-8 -*-

import unittest
import datetime
import re
from datetime import datetime, timedelta
from textwrap import dedent

import sys
import os
sys.path.append(os.path.realpath(__file__ + '/../../../lib'))
import udf
import exatest

from vschema_common import VSchemaTest, TestUtils

# Assumptions
# - JDBC Adapter correct for EXASolution remote database (except pushdown, which is not used here).
# 
# 
# Virtual Schema
# Corner Cases
# - Huge adapaterNotes => Add test
# - Many tables/columns (long json) => Test how much is possible

#@unittest.skip("skipped test")
class CreateVirtualSchemaTest(VSchemaTest):

    setupDone = False

    def setUp(self):
        # TODO Remove this workaround
        if self.__class__.setupDone:
            self.query(''' OPEN SCHEMA VS1 ''')
            return
            
        # Create a simple native schema with tables
        
        self.query('DROP SCHEMA IF EXISTS NATIVE CASCADE')
        self.query('CREATE SCHEMA NATIVE')
        self.query('CREATE TABLE T1(a int, b varchar(100), c double)')
        self.query('CREATE TABLE T2(a date, b timestamp, c boolean)')
        self.query('CREATE TABLE T3(c1 char, c2 decimal(18,5))')
        self.query('CREATE TABLE T4(c1 integer identity)')
        self.query('CREATE TABLE T5(c1 int identity)')
        self.query('CREATE TABLE T6(c1 smallint identity)')
        self.query('CREATE TABLE T7(c1 decimal(5,0) identity)')
        self.query('''CREATE TABLE T8(c1 boolean default TRUE, c2 char(10) default 'foo', c3 date default '2016-06-01', c4 decimal(5,0) default 0)''')
        self.query('''CREATE TABLE T9(c1 double default 1E2, c2 geometry default 'POINT(2 5)', c3 interval year to month default '3-5', c4 interval day to second default '2 12:50:10.123')''')
        self.query('''CREATE TABLE TA(c1 timestamp default '2016-06-01 00:00:01.000', c2 timestamp with local time zone default '2016-06-01 00:00:02.000', c3 varchar(100) default 'bar')''')
        self.query('''CREATE TABLE TB(c1 boolean default NULL, c2 char(10) default NULL, c3 date default NULL, c4 decimal(5,0) default NULL)''')
        self.query('''CREATE TABLE TC(c1 double default NULL, c2 geometry default NULL, c3 interval year to month default NULL, c4 interval day to second default NULL)''')
        self.query('''CREATE TABLE TD(c1 timestamp default NULL, c2 timestamp with local time zone default NULL, c3 varchar(100) default NULL)''')
        self.query('''CREATE TABLE TE(c1 integer comment is '', c2 integer comment is 'This is a comment.')''')
        self.query('''CREATE TABLE TF(c1 integer NOT NULL, c2 varchar(100) NOT NULL, c3 double NOT NULL)''')
        self.query('DROP SCHEMA IF EXISTS NATIVE2 CASCADE')
        self.query('CREATE SCHEMA NATIVE2')
        self.commit()  # commit, otherwise adapter doesn't see tables
        
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS2", "NATIVE2", "ADAPTER.JDBC_ADAPTER", True)
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        
        self.__class__.setupDone = True

    def testCurrentSchema(self):
        rows = self.query(''' SELECT CURRENT_SCHEMA ''')
        self.assertRowsEqual([("VS1",)], rows)

    def testCAT(self):
        rows = self.query(''' SELECT * FROM CAT order by table_name ''')
        self.assertRowsEqual([("T1","TABLE"),("T2","TABLE"),("T3","TABLE"),
            ("T4","TABLE"),("T5","TABLE"),("T6","TABLE"),("T7","TABLE"),
            ("T8","TABLE"),("T9","TABLE"),("TA","TABLE"),("TB","TABLE"),
            ("TC","TABLE"),("TD","TABLE"),("TE","TABLE"),("TF","TABLE")], rows)

    def testDescribe(self):
        for tableName in ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "TA", "TB", "TC", "TD", "TE", "TF"]:
            rows = self.query(''' DESCRIBE {t} '''.format(t = tableName))
            self.assertEqual(['COLUMN_NAME', 'SQL_TYPE', 'NULLABLE', 'DISTRIBUTION_KEY'], self.columnNames())
            rows_native = self.query(''' DESCRIBE native.{t} '''.format(t = tableName))
            self.assertEqual(self.getColumn(rows_native,0), self.getColumn(rows,0))
            # GD201606: TODO: This is only a workaround for the GEOMETRY Column Type
            if (tableName == "T9" or tableName == "TC"):
                self.assertEqual(['DOUBLE', 'GEOMETRY(3857)', 'INTERVAL YEAR(2) TO MONTH','INTERVAL DAY(2) TO SECOND(3)'], self.getColumn(rows,1))
            else:
                self.assertEqual(self.getColumn(rows_native,1), self.getColumn(rows,1))
            # nullable and distributionkey column should be NULL
            self.assertColumnEqualConst(rows, 2, None)
            self.assertColumnEqualConst(rows, 3, None)
        
    def testSysTableSchemas(self):
        rows = self.query('''
            SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(['SCHEMA_NAME', 'SCHEMA_OWNER', 'SCHEMA_OBJECT_ID', 'SCHEMA_IS_VIRTUAL', 'SCHEMA_COMMENT'], self.columnNames())
        self.assertEqual(1, self.rowcount())
        self.assertEqual("VS1", rows[0][0])
        self.assertEqual("SYS", rows[0][1])
        schemaObjectId = rows[0][2]
        self.assertTrue(schemaObjectId != None and schemaObjectId > 0)
        self.assertEqual(True, rows[0][3])
        self.assertEqual(None, rows[0][4])
        
        rows = self.query('''
            SELECT * FROM EXA_VIRTUAL_SCHEMAS WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(['SCHEMA_NAME', 'SCHEMA_OWNER', 'SCHEMA_OBJECT_ID', 'ADAPTER_SCRIPT', 'LAST_REFRESH', 'LAST_REFRESH_BY', 'ADAPTER_NOTES'], self.columnNames())
        self.assertEqual(1, self.rowcount())
        self.assertEqual("VS1", rows[0][0])
        self.assertEqual("SYS", rows[0][1])
        self.assertEqual(schemaObjectId, rows[0][2])
        self.assertEqual("ADAPTER.JDBC_ADAPTER", rows[0][3])
        self.assertEqual("SYS", rows[0][5])
        # Check last refreshed. Take server time
        schemaLastRefreshed = rows[0][4]
        currentTime = self.queryCurrentTimestamp()
        diff = currentTime - schemaLastRefreshed
        self.assertGreaterEqual(diff.total_seconds(), 0)
        self.assertLess(diff.total_seconds(), 10)
        
    def testSysTableTables(self):
        rows = self.query('''
            SELECT * FROM EXA_DBA_TABLES WHERE TABLE_SCHEMA='VS1' ORDER BY TABLE_NAME
            ''')
        self.assertEqual(['TABLE_SCHEMA', 'TABLE_NAME', 'TABLE_OWNER', 'TABLE_OBJECT_ID', 'TABLE_IS_VIRTUAL', 'TABLE_HAS_DISTRIBUTION_KEY', 'TABLE_ROW_COUNT', 'DELETE_PERCENTAGE', 'TABLE_COMMENT'], self.columnNames())
        self.assertEqual(15, self.rowcount())
        self.assertColumnEqualConst(rows, 0, 'VS1')
        self.assertEqual(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "TA", "TB", "TC", "TD", "TE", "TF"], self.getColumn(rows,1))
        self.assertColumnEqualConst(rows, 2, 'SYS')
        for i in range(0,self.rowcount()):
            tableObjectId = rows[i][3]
            self.assertTrue(tableObjectId != None and tableObjectId > 0)
        self.assertColumnEqualConst(rows, 4, True)
        self.assertColumnEqualConst(rows, 5, None)
        self.assertColumnEqualConst(rows, 7, None)
        self.assertColumnEqualConst(rows, 8, None)
        
        rows = self.query('''
            SELECT * FROM EXA_DBA_VIRTUAL_TABLES WHERE TABLE_SCHEMA = 'VS1' ORDER BY TABLE_NAME
            ''')
        self.assertEqual(['TABLE_SCHEMA', 'TABLE_NAME', 'TABLE_OBJECT_ID', 'LAST_REFRESH', 'LAST_REFRESH_BY', 'ADAPTER_NOTES'], self.columnNames())
        self.assertEqual(15, self.rowcount())
        self.assertColumnEqualConst(rows, 0, 'VS1')
        self.assertEqual(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "TA", "TB", "TC", "TD", "TE", "TF"], self.getColumn(rows,1))
        for i in range(0,self.rowcount()):
            tableObjectId = rows[i][2]
            self.assertTrue(tableObjectId != None and tableObjectId > 0)
            # check last refreshed
            tableLastRefreshed = rows[i][3]
            diff = tableLastRefreshed - self.getLastSchemaRefresh('VS1')
            self.assertGreaterEqual(diff.total_seconds(), 0)
            self.assertLess(diff.total_seconds(), 10)
        self.assertColumnEqualConst(rows, 4, 'SYS')
        
    def testSysTableColumns(self):
        rows = self.query('''
            SELECT * FROM EXA_DBA_COLUMNS WHERE COLUMN_SCHEMA = 'VS1' ORDER BY COLUMN_TABLE, COLUMN_ORDINAL_POSITION
            ''')
        expectedCols = ["COLUMN_SCHEMA", "COLUMN_TABLE", "COLUMN_OBJECT_TYPE", "COLUMN_NAME", "COLUMN_TYPE", "COLUMN_TYPE_ID",
            "COLUMN_MAXSIZE", "COLUMN_NUM_PREC", "COLUMN_NUM_SCALE", "COLUMN_ORDINAL_POSITION", "COLUMN_IS_VIRTUAL",
            "COLUMN_IS_NULLABLE", "COLUMN_IS_DISTRIBUTION_KEY", "COLUMN_DEFAULT", "COLUMN_IDENTITY", "COLUMN_OWNER", "COLUMN_OBJECT_ID",
            "STATUS", "COLUMN_COMMENT"]
        self.assertEqual(expectedCols, self.columnNames())
        self.assertEqual(39, self.rowcount())
        rows_native = self.query('''
            SELECT * FROM EXA_DBA_COLUMNS WHERE COLUMN_SCHEMA = 'VS1' ORDER BY COLUMN_TABLE, COLUMN_ORDINAL_POSITION
            ''')
        self.assertEqual(expectedCols, self.columnNames())
        self.assertEqual(39, self.rowcount())
        for i in range (1,10):
            self.assertEqual(self.getColumn(rows_native,i), self.getColumn(rows,i))
        self.assertColumnEqualConst(rows, 10, True)
        for i in range (11,13):
            self.assertColumnEqualConst(rows, i, None)
        self.assertEqual(self.getColumn(rows_native,13), self.getColumn(rows,13))
        self.assertEqual(self.getColumn(rows_native,14), self.getColumn(rows,14))
        self.assertColumnEqualConst(rows, 15, 'SYS')
        for i in range(0,self.rowcount()):
            colObjectId = rows[i][16]
            self.assertTrue(colObjectId != None and colObjectId > 0)
        self.assertColumnEqualConst(rows, 17, None)
        self.assertEqual(self.getColumn(rows_native,18), self.getColumn(rows,18))

        rows = self.query('''
            SELECT * FROM EXA_DBA_VIRTUAL_COLUMNS WHERE COLUMN_SCHEMA = 'VS1' ORDER BY COLUMN_TABLE, COLUMN_NAME
            ''')
        self.assertEqual(["COLUMN_SCHEMA", "COLUMN_TABLE", "COLUMN_NAME", "COLUMN_OBJECT_ID", "ADAPTER_NOTES"], self.columnNames())
        self.assertEqual(39, self.rowcount())
        self.assertColumnEqualConst(rows, 0, 'VS1')
        rows_native = self.query('''
            SELECT COLUMN_TABLE, COLUMN_NAME FROM EXA_DBA_COLUMNS WHERE COLUMN_SCHEMA='NATIVE' ORDER BY COLUMN_TABLE, COLUMN_NAME
            ''')
        self.assertEqual(self.getColumn(rows_native,0), self.getColumn(rows,1))
        self.assertEqual(self.getColumn(rows_native,1), self.getColumn(rows,2))
        for i in range(0,self.rowcount()):
            colObjectId = rows[i][3]
            self.assertTrue(colObjectId != None and colObjectId > 0)
        # self.assertEqual([None]*self.rowcount(), self.getColumn(rows,4))
        
    def testSysTableSchemaProperties(self):
        rows = self.query('''
            SELECT * FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME='VS1' ORDER BY PROPERTY_NAME
            ''')
        self.assertEqual(["SCHEMA_NAME", "SCHEMA_OBJECT_ID", "PROPERTY_NAME", "PROPERTY_VALUE"], self.columnNames())
        self.assertEqual(7, self.rowcount())
        self.assertColumnEqualConst(rows, 0, 'VS1')
        for i in range(0,self.rowcount()):
            colObjectId = rows[i][1]
            self.assertTrue(colObjectId != None and colObjectId > 0)
        self.assertEqual(["CONNECTION_STRING", "EXCEPTION_HANDLING","IS_LOCAL", "PASSWORD", "SCHEMA_NAME", "SQL_DIALECT", "USERNAME"], self.getColumn(rows,2))
        self.assertEqual(["jdbc:exa:{hostport}".format(hostport = udf.opts.server), "NONE", "True", "exasol", "NATIVE", "EXASOL", "sys"], self.getColumn(rows,3))
        
    def testSysTableObjects(self):
        rows = self.query('''
            SELECT OBJECT_NAME, OBJECT_TYPE from EXA_DBA_OBJECTS WHERE OBJECT_IS_VIRTUAL = true AND ROOT_NAME = 'VS1' ORDER BY OBJECT_NAME
            ''')
        self.assertEqual(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "TA", "TB", "TC", "TD", "TE", "TF"], self.getColumn(rows,0))
        self.assertColumnEqualConst(rows, 1, 'TABLE')

    def testCreateWithConnection(self):
        # Create a Virtual Schema, now using a connection (will drop the existing virtual schema w/o connection)
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True, useConnection=True)
        rows = self.query('''
            SELECT OBJECT_NAME, OBJECT_TYPE from EXA_DBA_OBJECTS WHERE OBJECT_IS_VIRTUAL = true AND ROOT_NAME = 'VS1' ORDER BY OBJECT_NAME
            ''')
        self.assertEqual(["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "TA", "TB", "TC", "TD", "TE", "TF"], self.getColumn(rows,0))
        self.assertColumnEqualConst(rows, 1, 'TABLE')

    def testNotNull(self):
        # Not null is currently not carried over to the virtual schema.
        rows = self.query('''
            SELECT COLUMN_IS_NULLABLE FROM EXA_ALL_COLUMNS WHERE COLUMN_TABLE='TF' AND COLUMN_SCHEMA='NATIVE';
            ''')
        self.assertEqual([False, False, False], self.getColumn(rows,0))
        rows = self.query('''
            SELECT COLUMN_IS_NULLABLE FROM EXA_ALL_COLUMNS WHERE COLUMN_TABLE='TF' AND COLUMN_SCHEMA='VS1';
            ''')
        self.assertEqual([None, None, None], self.getColumn(rows,0))

    def testEmptySchema(self):
        # Test JDBC adapter on empty schema
        with self.assertRaisesRegexp(Exception, '''object VS2.DUMMY not found'''):
            rows = self.query("SELECT * FROM VS2.DUMMY")

    def testQuotedSchemaNames(self):
        self.createFastAdapter(schemaName='''"quoted_adapter"''', adapterName='''"fast_adapter"''')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        with self.assertRaisesRegexp(Exception, '''Could not find adapter script'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING quoted_adapter.fast_adapter ''')
        with self.assertRaisesRegexp(Exception, '''Could not find adapter script'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING quoted_adapter."fast_adapter" ''')
        with self.assertRaisesRegexp(Exception, '''Could not find adapter script'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING "quoted_adapter".fast_adapter ''')
        self.createFastAdapter(schemaName='''quoted_adapter''', adapterName='''"fast_adapter"''')
        self.createFastAdapter(schemaName='''"quoted_adapter2"''', adapterName='''fast_adapter''')
        self.query('''CREATE VIRTUAL SCHEMA VS1 USING "quoted_adapter"."fast_adapter" ''')
        rows = self.query('''
            SELECT * from VS1.DUMMY
            ''')
        self.assertRowsEqual([('FOO', 'BAR')],rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('''CREATE VIRTUAL SCHEMA VS1 USING "quoted_adapter2".fast_adapter ''')
        rows = self.query('''
            SELECT * from VS1.DUMMY
            ''')
        self.assertRowsEqual([('FOO', 'BAR')],rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('''CREATE VIRTUAL SCHEMA VS1 USING quoted_adapter."fast_adapter" ''')
        rows = self.query('''
            SELECT * from VS1.DUMMY
            ''')
        self.assertRowsEqual([('FOO', 'BAR')],rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        #self.query('''DROP SCHEMA IF EXISTS "quoted_adapter" CASCADE''')
        #self.query('''DROP SCHEMA IF EXISTS quoted_adapter CASCADE''')




class CreateForceVirtualSchemaTest(VSchemaTest):

    def testCreateForceVirtualSchema(self):
        self.createFailingAdapter(schemaName="ADAPTER", adapterName="FAILING_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE FORCE VIRTUAL SCHEMA VS1 USING ADAPTER.FAILING_ADAPTER')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(1, self.rowcount())
        rows = self.query('''
            SELECT * from EXA_DBA_OBJECTS WHERE OBJECT_IS_VIRTUAL = true AND ROOT_NAME = 'VS1'
            ''')
        self.assertEqual(0, self.rowcount())

class CreateVirtualSchemaWithProperties(VSchemaTest):

    def testWithNormalValue(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = 'default' ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)

    def testPropertyValueEmptyString(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        with self.assertRaisesRegexp(Exception, '''Value of property UNUSED must not be null or empty.'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = '' ''')

    def testPropertyValueNull(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        with self.assertRaisesRegexp(Exception, '''Value of property UNUSED must not be null or empty.'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = null ''')

    def testDuplicate(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        with self.assertRaisesRegexp(Exception, '''Duplicate property names \(UNUSED\) are not allowed.'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = 'default1' UNUSED = 'default2' ''')

    def testDuplicateWithEmptyString(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        with self.assertRaisesRegexp(Exception, '''Value of property UNUSED must not be null or empty.'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = 'default1' UNUSED = '' ''')

    def testDuplicateWithNull(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        with self.assertRaisesRegexp(Exception, '''Value of property UNUSED must not be null or empty.'''):
            self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = 'default1' UNUSED = null ''')

#@unittest.skip("skipped test")
class UnicodeAndCaseSensitivityTest(VSchemaTest):

    def setUp(self):
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")
        self.query('DROP SCHEMA IF EXISTS NATIVE CASCADE')
        self.query('CREATE SCHEMA NATIVE')
        # 茶 is 3-byte and ¥ is 2 byte in utf-8.
        self.query(u'''
            CREATE OR REPLACE TABLE "¥tAbLe"("a茶A" double, "b¥B" varchar(3)) ''')
        self.query(u'''
            INSERT INTO "¥tAbLe" VALUES
            (1.1, 'v茶V'),
            (2.2, 'v¥V') ''')
        self.commit()
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)  # TODO Make this IS_LOCAL = False as soon as the udf_wrapper.tcs file supports etlJdbcConfigDir
        self.commit()

    def test(self):
        # Create with IS_LOCAL true, because the EXASolution jdbc driver returns too big integer types which are converted
        rows = self.queryUnicode(u'''
            SELECT "a茶A", "b¥B" FROM VS1."¥tAbLe" ORDER BY 1 ''')
        # Bug: Column names are requested from pyodbc as UTF-16 or so (according to MT), but 茶 doesn't fit into two bytes.
        #self.assertEqual([u"a茶A", u"b¥B"], self.columnNames())
        self.assertRowsEqual([(1.1, u'v茶V'), (2.2, u'v¥V')],rows)
        # Join system tables to make sure that everything is consistent
        rows = self.decodeUtf8Fields(self.queryColumnMetadata('VS1'))
        self.assertEqual([u"¥tAbLe", u"¥tAbLe"], self.getColumnByName(rows, 'TABLE_NAME'))
        self.assertEqual([u"a茶A", u"b¥B"], self.getColumnByName(rows, 'COLUMN_NAME'))
        
        # Test special characters in properties
        with self.assertRaisesRegexp(Exception, 'Quoted property names and special characters in the property name are not allowed'):
            self.query(u'''
                ALTER VIRTUAL SCHEMA VS1 SET "Foo茶¥Bar" = 'v茶V¥v'
                ''')
        with self.assertRaisesRegexp(Exception, '''character is not allowed within property names'''):
            self.query(u'''
                ALTER VIRTUAL SCHEMA VS1 SET A.B = 'v茶V¥v'
                ''')
        self.query(u'''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'v茶V¥v'
            ''')
        propValue = self.queryScalarUnicode(u'''
            SELECT PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1' AND PROPERTY_NAME = 'UNUSED'
            ''')
        self.assertEqual(u'v茶V¥v', propValue)



class RefreshTest(VSchemaTest):

    def setUp(self):
        # Create a simple native schema with tables
        self.query('DROP SCHEMA IF EXISTS NATIVE CASCADE')
        self.query('CREATE SCHEMA NATIVE')
        self.query('CREATE TABLE T1(a int, b varchar(100))')
        self.query('CREATE TABLE T2(c date)')
        self.query('CREATE TABLE T3(d double)')
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()

    def testRefreshAll(self):
        # Refresh all (new/changed/deleted tables). Metadata for all tables are rewritten.
        self.query('DROP TABLE NATIVE.T3')
        self.query('ALTER TABLE NATIVE.T2 ADD COLUMN d int')
        self.query('CREATE TABLE NATIVE.T4(a int)')
        self.commit()
        timeBefore = self.queryCurrentTimestamp()
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 REFRESH ''')
        self.commit()
        timeAfter = self.queryCurrentTimestamp()
        # Check last_refreshed
        self.assertBetween(self.getLastTableRefresh ('VS1', 'T1'), timeBefore, timeAfter)
        self.assertBetween(self.getLastTableRefresh ('VS1', 'T2'), timeBefore, timeAfter)
        # check metadata
        rows = self.queryColumnMetadata('VS1')
        self.assertRowsEqual([('T1', 'A', 'DECIMAL(18,0)'), ('T1', 'B', 'VARCHAR(100) UTF8'), ('T2', 'C', 'DATE'), ('T2', 'D', 'DECIMAL(18,0)'), ('T4', 'A', 'DECIMAL(18,0)')], rows)

    def testRefreshTables(self):
        # Remember date from T1 for later
        table1Before = self.getLastTableRefresh('VS1', 'T1')
        # Refresh specific Tables
        self.query('ALTER TABLE NATIVE.T1 ADD COLUMN e int')    # this should be ignored during the refresh
        self.query('DROP TABLE NATIVE.T2')
        self.query('ALTER TABLE NATIVE.T3 ADD COLUMN f int')
        self.query('CREATE TABLE NATIVE.T4(e double)')
        # Now we have T1, T3, T4
        self.commit()
        timeBefore = self.queryCurrentTimestamp()
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 REFRESH TABLES T2 T3 T4 ''')
        self.commit()
        timeAfter = self.queryCurrentTimestamp()
        # check last refresh
        self.assertBetween(self.getLastSchemaRefresh('VS1'), timeBefore, timeAfter)
        self.assertEqual  (self.getLastTableRefresh ('VS1', 'T1'), table1Before)   # t1 should be unchanged
        self.assertBetween(self.getLastTableRefresh ('VS1', 'T3'), timeBefore, timeAfter)
        self.assertBetween(self.getLastTableRefresh ('VS1', 'T4'), timeBefore, timeAfter)
        # check metadata
        rows = self.queryColumnMetadata('VS1')
        self.assertRowsEqual([('T1', 'A', 'DECIMAL(18,0)'), ('T1', 'B', 'VARCHAR(100) UTF8'), ('T3', 'D', 'DOUBLE'), ('T3', 'F', 'DECIMAL(18,0)'), ('T4', 'E', 'DOUBLE')], rows)

class SetPropertiesTest(VSchemaTest):

    def testCreateProperty(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query(u'''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default'
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)


    def testChangeProperty(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default'
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'newValue'
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'newValue')],rows)

    def testChangePropertyFromCreate(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = 'default' ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'newValue'
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'newValue')],rows)

    def testDeleteNonExistingPropertyWithEmptyString(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = ''
            ''')
        self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(0, self.rowcount())

    def testDeleteNonExistingPropertyWithNull(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = NULL
            ''')
        self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(0, self.rowcount())

    def testDeletePropertyWithEmptyString(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default'
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = ''
            ''')
        self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(0, self.rowcount())

    def testDeletePropertyWithNull(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default'
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = NULL
            ''')
        self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(0, self.rowcount())

    def testDeletePropertyFromCreateWithEmptyString(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = 'default' ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = ''
            ''')
        self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(0, self.rowcount())

    def testDeletePropertyFromCreateWithNull(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('''CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH UNUSED = 'default' ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = NULL
            ''')
        self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertEqual(0, self.rowcount())

    def testDeletePropertyTwice(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('''
           ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default'
           ''')
        rows = self.query('''
           SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
           ''')
        self.assertRowsEqual([('UNUSED', 'default')],rows)
        self.query('''
           ALTER VIRTUAL SCHEMA VS1 SET UNUSED = NULL
           ''')
        self.query('''
           SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
           ''')
        self.assertEqual(0, self.rowcount())
        self.query('''
           ALTER VIRTUAL SCHEMA VS1 SET UNUSED = NULL
           ''')
        self.query('''
           SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
           ''')
        self.assertEqual(0, self.rowcount())

    def testDeleteOnlyOneProperty(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default' UNUSED2 = 'default2'
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1' ORDER BY PROPERTY_NAME
            ''')
        self.assertRowsEqual([('UNUSED', 'default'), ('UNUSED2', 'default2')],rows)
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET UNUSED2 = 'Not deleted' UNUSED = NULL
            ''')
        rows = self.query('''
            SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'
            ''')
        self.assertRowsEqual([('UNUSED2', 'Not deleted')],rows)

    def testDuplicatePropertyName(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        with self.assertRaisesRegexp(Exception, 'Duplicate property names \\(UNUSED\\) are not allowed.'):
            self.query('''
                ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default' UNUSED = 'default2'
                ''')
        with self.assertRaisesRegexp(Exception, 'Duplicate property names \\(UNUSED\\) are not allowed.'):
            self.query('''
                ALTER VIRTUAL SCHEMA VS1 SET UNUSED = null UNUSED = 'default2'
                ''')
        with self.assertRaisesRegexp(Exception, 'Duplicate property names \\(UNUSED\\) are not allowed.'):
            self.query('''
                ALTER VIRTUAL SCHEMA VS1 SET UNUSED = 'default' UNUSED = null
                ''')
        with self.assertRaisesRegexp(Exception, 'Duplicate property names \\(UNUSED\\) are not allowed.'):
            self.query('''
                ALTER VIRTUAL SCHEMA VS1 SET UNUSED = null UNUSED = null
                ''')

    def testOldPropertiesInSchemaMetadataInfo(self):
        self.createTestPropertyAdapter(schemaName="ADAPTER", adapterName="TEST_PROPERTY_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.TEST_PROPERTY_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET P1='1' P2='2'
            ''')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET P1='1new' P2=null P3='3'
            ''')

    def testInvalidPropertiesInSchemaMetadataInfo(self):
        # Invalid properties => Add test with custom adapter for invalid properties displaying correct error message
        self.createTestPropertyAdapter(schemaName="ADAPTER", adapterName="TEST_PROPERTY_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.TEST_PROPERTY_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET P2='2'
            ''')
        with self.assertRaisesRegexp(Exception, 'Expected different values for old properties'):
            self.query('''
                ALTER VIRTUAL SCHEMA VS1 SET P1='1'
                ''')
        with self.assertRaisesRegexp(Exception, 'Expected different values for old properties'):
            self.query('''
                ALTER VIRTUAL SCHEMA VS1 SET P1='42' P2='2'
                ''')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.TEST_PROPERTY_ADAPTER')
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET P1='1' P2='2'
            ''')
        with self.assertRaisesRegexp(Exception, 'Expected different values for new properties'):
            self.query('''
                ALTER VIRTUAL SCHEMA VS1 SET P1=null P2=null P3='4'
                ''')

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
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).encode('utf-8')
                elif root["type"] == "dropVirtualSchema":
                    return json.dumps({{"type": "dropVirtualSchema"}}).encode('utf-8')
                elif root["type"] == "setProperties":
                    expectedOldProperties = {{'P1': '1', 'P2': '2'}}
                    expectedNewProperties = {{'P1': '1new', 'P2': None,'P3': '3'}}
                    if (root["schemaMetadataInfo"].get("properties", None) != None and len(root["schemaMetadataInfo"]["properties"]) > 0):
                        assert (len(root["schemaMetadataInfo"]["properties"]) == len(expectedOldProperties)), 'Expected different values for old properties. Expected: ' + str(expectedOldProperties) + ' Actual: ' + str(root["schemaMetadataInfo"]["properties"])
                        for propertyName, propertyValue in root["schemaMetadataInfo"]["properties"].iteritems():
                            assert (propertyName in expectedOldProperties), 'Expected different values for old properties. Expected: ' + str(expectedOldProperties) + ' actual: ' + str(root["schemaMetadataInfo"]["properties"])
                            assert (propertyValue == expectedOldProperties.get(propertyName, None)), 'Expected different values for old properties. Expected: ' + str(expectedOldProperties) + ' Actual: ' + str(root["schemaMetadataInfo"]["properties"])
                        assert (len(root["properties"]) == len(expectedNewProperties)), 'Expected different values for new properties. Expected: ' +  str(expectedNewProperties) + ' Actual: ' + str(root["properties"])
                        for propertyName, propertyValue in root["properties"].iteritems():
                            assert (propertyName in expectedNewProperties), 'Expected different values for new properties. Expected: ' +  str(expectedNewProperties) + ' Actual: ' + str(root["properties"])
                            assert (propertyValue == expectedNewProperties.get(propertyName, None)), 'Expected different values for new properties. Expected: ' +  str(expectedNewProperties) + ' Actual: ' + str(root["properties"])
                    return json.dumps({{"type": "setProperties"}}).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

class SetPropertiesRefreshTest(VSchemaTest):
    
    def setUp(self):
        # Create a simple native schema with tables
        self.createNative()
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()

    def testWithRefresh(self):
        self.query('DROP TABLE NATIVE.T_DATETIME')
        self.query('ALTER TABLE NATIVE.T ADD COLUMN d int')
        self.query('CREATE TABLE NATIVE.T_NEW(a int)')
        self.query('DROP SCHEMA IF EXISTS NATIVE_RENAMED CASCADE')
        self.query('RENAME SCHEMA NATIVE TO NATIVE_RENAMED')
        self.commit()
        timeBefore = self.queryCurrentTimestamp()
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET SCHEMA_NAME='{remoteSchema}' CONNECTION_STRING='jdbc:exa:{host_port};schema={remoteSchema}'
            '''.format(host_port = udf.opts.server,remoteSchema='NATIVE_RENAMED'))
        self.commit()  # without this commit, the refresh time does not get updated
        timeAfter = self.queryCurrentTimestamp()
        rows = self.queryColumnMetadata('VS1')
        self.assertRowsEqual(
            [('G', 'K', 'DECIMAL(18,0)'), ('G', 'V1', 'DECIMAL(18,0)'), ('G', 'V2', 'VARCHAR(100) UTF8'),
            ('NUMBERS1', 'A', 'DECIMAL(18,0)'), ('NUMBERS1', 'B', 'DECIMAL(18,0)'), ('NUMBERS1', 'C', 'DECIMAL(18,0)'), ('NUMBERS1', 'D', 'DECIMAL(18,0)'),
            ('NUMBERS2', 'E', 'DECIMAL(18,0)'), ('NUMBERS2', 'F', 'DECIMAL(18,0)'), ('NUMBERS2', 'G', 'DECIMAL(18,0)'), ('NUMBERS2', 'H', 'DECIMAL(18,0)'),
            ('T', 'A', 'DECIMAL(18,0)'), ('T', 'B', 'VARCHAR(100) UTF8'), ('T', 'C', 'DOUBLE'), ('T', 'D', 'DECIMAL(18,0)'),
            ('TEST', 'A', 'TIMESTAMP WITH LOCAL TIME ZONE'),
            ('T_CONNECT', 'PARENT', 'DECIMAL(18,0)'),
            ('T_CONNECT', 'VAL', 'DECIMAL(18,0)'),
            ('T_DATATYPES', 'A1', 'DECIMAL(18,0)'),
            ('T_DATATYPES', 'A10', 'GEOMETRY(3857)'),
            ('T_DATATYPES', 'A11', 'DECIMAL(10,5)'),
            ('T_DATATYPES', 'A12', 'DOUBLE'),
            ('T_DATATYPES', 'A13', 'DECIMAL(36,0)'),
            ('T_DATATYPES', 'A14', 'DECIMAL(18,0)'),
            ('T_DATATYPES', 'A15', 'DECIMAL(29,0)'),
            ('T_DATATYPES', 'A16', 'DECIMAL(18,0)'),
            ('T_DATATYPES', 'A17', 'DECIMAL(25,0)'),
            ('T_DATATYPES', 'A18', 'DECIMAL(27,9)'),
            ('T_DATATYPES', 'A19', 'DOUBLE'),
            ('T_DATATYPES', 'A2', 'DOUBLE'),
            ('T_DATATYPES', 'A20', 'DECIMAL(18,0)'),
            ('T_DATATYPES', 'A21', 'DOUBLE'),
            ('T_DATATYPES', 'A22', 'DECIMAL(1,0)'),
            ('T_DATATYPES', 'A23', 'DECIMAL(3,2)'),
            ('T_DATATYPES', 'A24', 'DECIMAL(18,0)'),
            ('T_DATATYPES', 'A25', 'DECIMAL(6,0)'),
            ('T_DATATYPES', 'A26', 'DECIMAL(6,3)'),
            ('T_DATATYPES', 'A27', 'DOUBLE'),
            ('T_DATATYPES', 'A28', 'DECIMAL(9,0)'),
            ('T_DATATYPES', 'A29', 'DECIMAL(9,0)'),
            ('T_DATATYPES', 'A3', 'DATE'),
            ('T_DATATYPES', 'A30', 'DECIMAL(3,0)'),
            ('T_DATATYPES', 'A31', 'DATE'),
            ('T_DATATYPES', 'A32', 'TIMESTAMP WITH LOCAL TIME ZONE'),
            ('T_DATATYPES', 'A4', 'TIMESTAMP'),
            ('T_DATATYPES', 'A5', 'VARCHAR(3000) UTF8'),
            ('T_DATATYPES', 'A6', 'CHAR(10) UTF8'),
            ('T_DATATYPES', 'A7', 'BOOLEAN'),
            ('T_DATATYPES', 'A8', 'INTERVAL DAY(2) TO SECOND(3)'),
            ('T_DATATYPES', 'A9', 'INTERVAL YEAR(2) TO MONTH'),
            ('T_GEOMETRY', 'A', 'GEOMETRY(3857)'),
            ('T_GEOMETRY', 'ID', 'DECIMAL(18,0)'),
            ('T_INTERVAL', 'A', 'INTERVAL YEAR(2) TO MONTH'),
            ('T_INTERVAL', 'B', 'INTERVAL DAY(2) TO SECOND(3)'),
            ('T_NEW', 'A', 'DECIMAL(18,0)'),
            ('T_NULLS', 'A', 'DECIMAL(18,0)'),
            ('T_NULLS', 'B', 'VARCHAR(100) UTF8')], rows)
        # Check refresh time
        self.assertBetween(self.getLastSchemaRefresh('VS1'), timeBefore, timeAfter)
        self.assertBetween(self.getLastTableRefresh ('VS1', 'T'), timeBefore, timeAfter)
        self.assertBetween(self.getLastTableRefresh ('VS1', 'G'), timeBefore, timeAfter)
        self.assertBetween(self.getLastTableRefresh ('VS1', 'T_NEW'), timeBefore, timeAfter)

    def testWithoutRefresh(self):
        schemaRefreshBefore = self.getLastSchemaRefresh('VS1')
        tRefreshBefore      = self.getLastTableRefresh ('VS1', 'T')
        gRefreshBefore      = self.getLastTableRefresh ('VS1', 'G')
        tNewRefreshBefore   = self.getLastTableRefresh ('VS1', 'T_DATETIME')
        metaBefore = self.queryColumnMetadata('VS1')
        # Change the source schema
        self.query('DROP TABLE NATIVE.T_DATETIME')
        self.query('ALTER TABLE NATIVE.T ADD COLUMN d int')
        self.query('CREATE TABLE NATIVE.T_NEW(a int)')
        # Setting this property should not refresh tables
        timeBefore = self.queryCurrentTimestamp()
        self.query('''
            ALTER VIRTUAL SCHEMA VS1 SET IS_LOCAL='false'
            '''.format(host_port = udf.opts.server,remoteSchema='NATIVE_RENAMED'))
        self.commit()  # without this commit, the refresh time does not get updated
        timeAfter = self.queryCurrentTimestamp()
        self.assertBetween(self.getLastSchemaRefresh('VS1'), timeBefore, timeAfter)
        self.assertEqual  (self.getLastTableRefresh ('VS1', 'T'),          tRefreshBefore)
        self.assertEqual  (self.getLastTableRefresh ('VS1', 'G'),          gRefreshBefore)
        self.assertEqual  (self.getLastTableRefresh ('VS1', 'T_DATETIME'), tNewRefreshBefore)
        metaAfter = self.queryColumnMetadata('VS1')
        self.assertRowsEqual(metaBefore, metaAfter)


class DropVSchemaTest(VSchemaTest):
    
    def setUp(self):
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")

    def testDropVSchema(self):
        self.createNative()
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.query('DROP VIRTUAL SCHEMA VS1 CASCADE')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(0, self.rowcount())
        
    def testDropEmptyVSchema(self):
        self.query('DROP SCHEMA IF EXISTS NATIVE CASCADE')
        self.query('CREATE SCHEMA NATIVE')
        self.commit()
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.query('DROP VIRTUAL SCHEMA VS1')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(0, self.rowcount())
        
    def testDropVSchemaInvalidAdapterScript(self):
        self.createNative()
        self.commit()
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.JDBC_ADAPTER AS
            
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'SyntaxError: invalid syntax \\(JDBC_ADAPTER, line 1\\)'):
            self.query('DROP VIRTUAL SCHEMA VS1 CASCADE')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(1, self.rowcount())
        self.query('DROP FORCE VIRTUAL SCHEMA VS1 CASCADE')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(0, self.rowcount())
        
    def testDropVSchemaInvalidJson(self):
        self.createNative()
        self.commit()
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.JDBC_ADAPTER AS
            import json
            def adapter_call(request):
                # missing brackets
                return """ "type": "dropVirtualSchema"} """
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'Unknown exception while parsing the response: in Json::Value::find\\(key, end, found\\): requires objectValue or nullValue'):
            self.query('DROP VIRTUAL SCHEMA VS1 CASCADE')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(1, self.rowcount())
        self.query('DROP FORCE VIRTUAL SCHEMA VS1 CASCADE')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(0, self.rowcount())

    def testDropVSchemaMissingCascade(self):
        self.createNative()
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        with self.assertRaisesRegexp(Exception, 'schema is not empty - use DROP VIRTUAL SCHEMA VS1 CASCADE to delete it'):
            self.query('DROP VIRTUAL SCHEMA VS1')

    def testDropSchemaMissingVirtual(self):
        self.createNative()
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        with self.assertRaisesRegexp(Exception, 'schema VS1 is a virtual schema. Please use DROP VIRTUAL SCHEMA instead'):
            self.query('DROP SCHEMA VS1 CASCADE')

    def testDropForceVirtualSchema(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.FAST_ADAPTER AS
            import json
            import string
            def adapter_call(request):
                raise ValueError('This should never be called')
            /
            '''))
        self.query('DROP FORCE VIRTUAL SCHEMA VS1 CASCADE')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(0, self.rowcount())


class UnsupportedActions(VSchemaTest):

    setupDone = False

    def setUp(self):
        # TODO Remove this workaround
        if self.__class__.setupDone:
            self.query(''' OPEN SCHEMA VS1 ''')
            return
        # Create a simple native schema with tables
        self.createNative()
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.__class__.setupDone = True

    def testDDLCreateTable(self):
        with self.assertRaisesRegexp(Exception, 'Creating tables in virtual schemas is not allowed'):
            self.query('CREATE TABLE t_new (a int)')
        with self.assertRaisesRegexp(Exception, 'Creating tables in virtual schemas is not allowed'):
            self.query('CREATE TABLE t_new AS SELECT * FROM NATIVE.T')
        # SELECT INTO FROM is similar to CREATE TABLE AS
        with self.assertRaisesRegexp(Exception, 'Creating tables in virtual schemas is not allowed'):
            self.query('SELECT * INTO TABLE t_new FROM NATIVE.T;')
        with self.assertRaisesRegexp(Exception, 'Creating tables in virtual schemas is not allowed'):
            self.query('CREATE TABLE t_new LIKE NATIVE.T')

    def testDDLDropTable(self):
        with self.assertRaisesRegexp(Exception, 'Dropping virtual tables is not allowed'):
            self.query('DROP TABLE t')

    def testDDLAlterTableColumn(self):
        alterError = 'Altering virtual tables is not allowed'
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t ADD COLUMN new_col int')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t DROP COLUMN a')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t MODIFY COLUMN a double')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t RENAME COLUMN a TO x')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t ALTER COLUMN a SET DEFAULT 1')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t ALTER COLUMN a DROP DEFAULT')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t ALTER COLUMN a SET IDENTITY')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t ALTER COLUMN a DROP IDENTITY')
            
    def testDDLAlterTableDistribution(self):
        alterError = 'Altering virtual tables is not allowed'
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t DISTRIBUTE BY a')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t DROP DISTRIBUTION KEYS')
            
    def testDDLAlterTableConstraints(self):
        alterError = 'Altering virtual tables is not allowed'
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t ADD CONSTRAINT my_prim_key PRIMARY KEY (a)')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t MODIFY CONSTRAINT my_constraint DISABLE')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t MODIFY PRIMARY KEY DISABLE')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t DROP CONSTRAINT my_constraint')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t DROP PRIMARY KEY')
        with self.assertRaisesRegexp(Exception, alterError):
            self.query('ALTER TABLE t RENAME CONSTRAINT my_constraint TO my_constraint_new')

    def testDDLAlterTableForeignKey(self):
        with self.assertRaisesRegexp(Exception, 'references to virtual tables are not supported'):
            self.query('ALTER TABLE native.t ADD CONSTRAINT foreign key (a) REFERENCES vs1.t (a)')

    def testDDLCreateView(self):
        with self.assertRaisesRegexp(Exception, 'Creating views in virtual schemas is not allowed'):
            self.query('CREATE VIEW new_view AS SELECT * FROM native.t')
        with self.assertRaisesRegexp(Exception, 'Creating views in virtual schemas is not allowed'):
            self.query('CREATE FORCE VIEW new_view AS SELECT * FROM native.t')
            
    def testDDLDropView(self):
        with self.assertRaisesRegexp(Exception, 'view NON_EXISTING_VIEW does not exist'):
            self.query('DROP VIEW non_existing_view')
            
    def testDDLCreateFunction(self):
        with self.assertRaisesRegexp(Exception, 'Creating functions in virtual schemas is not allowed'):
            self.query(udf.fixindent('''
            CREATE OR REPLACE FUNCTION my_fun (a DECIMAL)
                RETURN VARCHAR(10)
                IS
                    res VARCHAR(10);
                BEGIN
                    res := 'foo';
                    RETURN res;
                END my_fun;
            /
            '''))

    def testDDLDropFunction(self):
        with self.assertRaisesRegexp(Exception, 'function NON_EXISTING_FUNC does not exist'):
            self.query('DROP FUNCTION non_existing_func')
            
    def testDDLCreateScriptingScript(self):
        with self.assertRaisesRegexp(Exception, 'Creating scripts in virtual schemas is not allowed'):
            self.query(udf.fixindent('''
            CREATE SCRIPT SCRIPT_B AS
            output("foo");
            /
            '''))
            
    def testDDLCreateUdfScript(self):
        with self.assertRaisesRegexp(Exception, 'Creating scripts in virtual schemas is not allowed'):
            self.query(udf.fixindent('''
            CREATE or replace PYTHON SET SCRIPT dummy(a int)
            EMITS (a int) AS
              def run(ctx):
                ctx.emit(1)
            /
            '''))
            
    def testDDLDropScript(self):
        with self.assertRaisesRegexp(Exception, 'script NON_EXISTING_SCRIPT does not exist'):
            self.query('DROP SCRIPT non_existing_script')
            
    def testDDLRenameObject(self):
        with self.assertRaisesRegexp(Exception, 'Renaming virtual schema objects is not allowed'):
            self.query('RENAME TABLE t TO t_new')
        with self.assertRaisesRegexp(Exception, 'object NON_EXISTING_VIEW does not exist'):
            self.query('RENAME VIEW non_existing_view TO view_new')
            
    def testDDLComment(self):
        with self.assertRaisesRegexp(Exception, 'Creating comments for virtual tables is not allowed'):
            self.query("COMMENT ON TABLE t IS 'table comment' ")
        with self.assertRaisesRegexp(Exception, 'Creating comments for virtual tables is not allowed'):
            self.query("COMMENT ON TABLE t IS 'table comment' (a IS 'col comment')")
        with self.assertRaisesRegexp(Exception, 'Creating column comments for virtual tables is not allowed'):
            self.query("COMMENT ON COLUMN t.a IS 'col comment'")
        with self.assertRaisesRegexp(Exception, 'script NON_EXISTING_SCRIPT not found'):
            self.query("COMMENT ON SCRIPT non_existing_script IS 'comment'")
            
    def testDMLInsert(self):
        with self.assertRaisesRegexp(Exception, 'Inserting into virtual tables is not supported'):
            self.query("INSERT INTO t values (1,'a',1)")
        with self.assertRaisesRegexp(Exception, 'Inserting into virtual tables is not supported'):
            self.query('INSERT INTO t DEFAULT VALUES')
        with self.assertRaisesRegexp(Exception, 'Inserting into virtual tables is not supported'):
            self.query('INSERT INTO t SELECT * FROM native.t')
            
    def testDMLUpdate(self):
        with self.assertRaisesRegexp(Exception, 'Updating virtual tables is not supported'):
            self.query('UPDATE t set a = 1')
        with self.assertRaisesRegexp(Exception, 'Updating virtual tables is not supported'):
            self.query('UPDATE t SET a = 1 WHERE a = 2')
        with self.assertRaisesRegexp(Exception, 'Updating virtual tables is not supported'):
            self.query('UPDATE t AS t1 SET t1.a = t2.a FROM t AS t2 WHERE t1.a = t2.a;')
    
    def testDMLMerge(self):
        with self.assertRaisesRegexp(Exception, 'cannot merge into a virtual table'):
            self.query('MERGE INTO t t1 USING native.t t2 ON (t1.a = t2.a) WHEN MATCHED THEN UPDATE SET a = t2.a')
        with self.assertRaisesRegexp(Exception, 'cannot use virtual table T2 as source for merge'):
            self.query('MERGE INTO native.t t1 USING t t2 ON (t1.a = t2.a) WHEN MATCHED THEN UPDATE SET b = t2.b')
            
    def testDMLDelete(self):
        with self.assertRaisesRegexp(Exception, 'Deleting from virtual tables is not supported'):
            self.query('DELETE FROM t WHERE a = 1')
        with self.assertRaisesRegexp(Exception, 'Truncating virtual tables is not supported'):
            self.query('DELETE FROM t')
            
    def testDMLTruncate(self):
        with self.assertRaisesRegexp(Exception, 'Truncating virtual tables is not supported'):
            self.query('TRUNCATE TABLE t')

    def testDMLImport(self):
        with self.assertRaisesRegexp(Exception, 'Inserting into virtual tables is not supported'):
            self.query("IMPORT INTO t FROM jdbc at 'jdbc:exa:invalid-host:5555' user 'sys' identified by 'exasol' statement 'SELECT * FROM NATIVE.T'")
        with self.assertRaisesRegexp(Exception, 'virtual tables cannot be used as error table'):
            self.query("IMPORT INTO native.t FROM jdbc at 'jdbc:exa:invalid-host:5555' user 'sys' identified by 'exasol' statement 'SELECT * FROM NATIVE.T' ERRORS INTO t")
        self.query('CREATE OR REPLACE TABLE native.t_copy LIKE native.t')
        with self.assertRaisesRegexp(Exception, re.escape('''Execution of SQL Statement (for reading data) failed on external EXASolution. [IMPORT directly from a virtual table is not supported. Use STATEMENT option instead with SELECT * FROM "VS1"."T" ''')):
            self.query("IMPORT INTO native.t_copy FROM exa at '{host_port}' user 'sys' identified by 'exasol' table VS1.T".format(host_port = udf.opts.server))
            
    def testDMLExport(self):
        self.query('CREATE OR REPLACE TABLE native.t_copy LIKE native.t')
        with self.assertRaisesRegexp(Exception, re.escape('''EXPORT directly from a virtual table is not supported. Use EXPORT (SELECT * FROM VS1.T) instead.''')):
            self.query("EXPORT VS1.T INTO JDBC at 'jdbc:exa:{host_port}' user 'sys' identified by 'exasol' TABLE native.t_copy".format(host_port = udf.opts.server))

    def testEnforceIndex(self):
        with self.assertRaisesRegexp(Exception, 'Enforcing indexes on virtual tables is not allowed'):
            self.query('ENFORCE INDEX ON t(a)')

    def testReorganize(self):
        with self.assertRaisesRegexp(Exception, 'Reorganizing a virtual table is not allowed'):
            self.query('REORGANIZE TABLE t')
        with self.assertRaisesRegexp(Exception, 'Reorganizing virtual schemas is not allowed'):
            self.query('REORGANIZE SCHEMA vs1')
        # Following query may not throw an exception
        self.query('REORGANIZE DATABASE')

    def testRecompress(self):
        with self.assertRaisesRegexp(Exception, 'Recompressing virtual tables is not allowed'):
            self.query('RECOMPRESS TABLE t')
        with self.assertRaisesRegexp(Exception, 'Recompressing virtual schemas is not allowed'):
            self.query('RECOMPRESS SCHEMA vs1')
        # Following query may not throw an exception
        self.query('RECOMPRESS DATABASE')

    def testPreload(self):
        with self.assertRaisesRegexp(Exception, 'Preloading a virtual table is not supported'):
            self.query('PRELOAD TABLE t')
        with self.assertRaisesRegexp(Exception, 'Preloading of virtual schemas is not supported'):
            self.query('PRELOAD SCHEMA vs1')
        # Following query may not throw an exception
        self.query('PRELOAD DATABASE')

    def testAnalyze(self):
        with self.assertRaisesRegexp(Exception, 'Analyzing statistics of virtual tables is not allowed'):
            self.query('ANALYZE TABLE t ESTIMATE STATISTICS')
        with self.assertRaisesRegexp(Exception, 'Analyzing statistics of virtual tables is not allowed'):
            self.query('ANALYZE TABLE t EXACT STATISTICS')
        with self.assertRaisesRegexp(Exception, 'Analyzing statistics of virtual schemas is not allowed'):
            self.query('ANALYZE SCHEMA vs1 ESTIMATE STATISTICS')
        # Following query may not throw an exception
        self.query('ANALYZE DATABASE ESTIMATE STATISTICS')

class AdapterScriptTest(VSchemaTest):

    def testDropAdapterWithExistingVSchema(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        with self.assertRaisesRegexp(Exception, 'At least one virtual schema existing for this Adapter Script, please drop all Virtual Schemas of this Adapter first'):
            self.query('DROP ADAPTER SCRIPT ADAPTER.FAST_ADAPTER')
        rows = self.query("SELECT * FROM EXA_SCHEMAS WHERE SCHEMA_NAME = 'VS1' ")
        self.assertEqual(1, self.rowcount())

    def testDropSchemaWithAdapterScriptsAndVSchemasNotAllowed(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        with self.assertRaisesRegexp(Exception, 'The schema contains the Adapter Script FAST_ADAPTER for which at least one Virtual Schema exists \(VS1\). Please drop all Virtual Schemas of this Adapter first.'):
            self.query('DROP SCHEMA ADAPTER CASCADE')
            
    def testDropSchemaWithAdapterScriptsButNoVSchemasAllowed(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP SCHEMA ADAPTER CASCADE')

    def testCreateVSchemaWithNonAdapterScriptFails(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT ADAPTER.ADAPTER_UDF (a int) EMITS (a varchar(100)) AS
            
            /
            '''))
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        with self.assertRaisesRegexp(Exception, 'Script ADAPTER.ADAPTER_UDF exists, but is not an Adapter Script as expected'):
            self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.ADAPTER_UDF')

    def testDropAdapterScriptFailsForNonAdapterScripts(self):
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON SET SCRIPT ADAPTER.ADAPTER_UDF (a int) EMITS (a varchar(100)) AS
            
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'script ADAPTER_UDF is not an adapter script. Please use DROP SCRIPT instead.'):
            self.query('DROP ADAPTER SCRIPT ADAPTER.ADAPTER_UDF')

    def testDropScriptFailsForAdapterScripts(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        with self.assertRaisesRegexp(Exception, 'script FAST_ADAPTER is an adapter script. Please use DROP ADAPTER SCRIPT instead.'):
            self.query('DROP SCRIPT ADAPTER.FAST_ADAPTER')

    def testOverwriteAdapterScriptByNonAdapterScriptFails(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        with self.assertRaisesRegexp(Exception, ''):
            self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON SET SCRIPT ADAPTER.FAST_ADAPTER (a int) EMITS (a varchar(100)) AS
                
                /
                '''))
        
    def testOverwriteNonAdapterScriptByAdapterScriptFails(self):
        self.query('DROP SCHEMA IF EXISTS ADAPTER_TMP CASCADE')
        self.query('CREATE SCHEMA ADAPTER_TMP')
        self.query(udf.fixindent('''
            CREATE PYTHON SET SCRIPT ADAPTER_TMP.MY_UDF (a int) EMITS (a varchar(100)) AS
            
            /
            '''))
        with self.assertRaisesRegexp(Exception, 'object MY_UDF already exists and is not an adapter script'):
            self.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER_TMP.MY_UDF AS
                
                /
                '''))

    def testOverwriteAdapterScriptByAnotherAdapter(self):
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.query('DROP SCHEMA IF EXISTS VS1 CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        rows = self.query("SELECT * FROM VS1.DUMMY")
        self.assertRowsEqual([('FOO', 'BAR')], rows)
        self.query(udf.fixindent('''
        CREATE OR REPLACE PYTHON ADAPTER SCRIPT adapter.fast_adapter AS
        import json
        import string
        def adapter_call(request):
            # database expects utf-8 encoded string of type str. unicode not yet supported
            root = json.loads(request)
            if root["type"] == "createVirtualSchema":
                res = {
                    "type": "createVirtualSchema",
                    "schemaMetadata": {
                        "tables": [
                        {
                            "name": "DUMMY",
                            "columns": [{
                                "name": "A",
                                "dataType": {"type": "VARCHAR", "size": 2000000}
                            },{
                                "name": "B",
                                "dataType": {"type": "VARCHAR", "size": 2000000}
                            }]
                        }]
                    }
                }
                return json.dumps(res).encode('utf-8')
            elif root["type"] == "dropVirtualSchema":
                return json.dumps({"type": "dropVirtualSchema"}).encode('utf-8')
            elif root["type"] == "setProperties":
                return json.dumps({"type": "setProperties"}).encode('utf-8')
            elif root["type"] == "refresh":
                return json.dumps({"type": "refresh"}).encode('utf-8')
            if root["type"] == "getCapabilities":
                return json.dumps({
                    "type": "getCapabilities",
                    "capabilities": []
                    }).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
            elif root["type"] == "pushdown":
                res = {
                    "type": "pushdown",
                    "sql": "SELECT * FROM (VALUES ('X', 'Y')) t"
                }
                return json.dumps(res).encode('utf-8')
            else:
                raise ValueError('Unsupported callback')
        /
            '''))
        rows = self.query("SELECT * FROM VS1.DUMMY")
        self.assertRowsEqual([('X', 'Y')], rows)

# Tests only the supported parts of IMPORT/EXPORT
class ImportExportTest(VSchemaTest):

    setupDone = False

    def setUp(self):
        # TODO Remove this workaround
        if self.__class__.setupDone:
            self.query(''' OPEN SCHEMA VS1 ''')
            return
        # Create a simple native schema with tables
        self.createNative()
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.__class__.setupDone = True

    def testExaImportFromQueryWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.g_copy LIKE native.g')
        self.query("IMPORT INTO native.g_copy FROM exa at '{host_port}' user 'sys' identified by 'exasol' statement 'SELECT * FROM vs1.g'".format(host_port = udf.opts.server))
        self.assertRowsEqualIgnoreOrder(
            self.query("SELECT * FROM native.g"),
            self.query("SELECT * FROM native.g_copy"))

    def testExaImportFromViewWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.g_copy LIKE native.g')
        self.query("CREATE OR REPLACE VIEW native.g_view AS SELECT * FROM vs1.g")
        self.commit()
        self.query("IMPORT INTO native.g_copy FROM exa at '{host_port}' user 'sys' identified by 'exasol' statement 'SELECT * FROM native.g_view'".format(host_port = udf.opts.server))
        self.assertRowsEqualIgnoreOrder(
            self.query("SELECT * FROM native.g"),
            self.query("SELECT * FROM native.g_copy"))

    def testJdbcImportFromVTable(self):
        self.query('CREATE OR REPLACE TABLE native.g_copy LIKE native.g')
        self.query("IMPORT INTO native.g_copy FROM jdbc at 'jdbc:exa:{host_port}' user 'sys' identified by 'exasol' TABLE vs1.g".format(host_port = udf.opts.server))
        self.assertRowsEqualIgnoreOrder(
            self.query("SELECT * FROM native.g"),
            self.query("SELECT * FROM native.g_copy"))

    def testJdbcImportFromQueryWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.g_copy LIKE native.g')
        self.query("IMPORT INTO native.g_copy FROM jdbc at 'jdbc:exa:{host_port}' user 'sys' identified by 'exasol' STATEMENT 'SELECT * FROM vs1.g'".format(host_port = udf.opts.server))
        self.assertRowsEqualIgnoreOrder(
            self.query("SELECT * FROM native.g"),
            self.query("SELECT * FROM native.g_copy"))

    def testJdbcImportFromViewWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.g_copy LIKE native.g')
        self.query("CREATE OR REPLACE VIEW native.g_view AS SELECT * FROM vs1.g")
        self.commit()
        self.query("IMPORT INTO native.g_copy FROM jdbc at 'jdbc:exa:{host_port}' user 'sys' identified by 'exasol' STATEMENT 'SELECT * FROM native.g_view'".format(host_port = udf.opts.server))
        self.assertRowsEqualIgnoreOrder(
            self.query("SELECT * FROM native.g"),
            self.query("SELECT * FROM native.g_copy"))

    def testExportFromQueryWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.g_copy LIKE native.g')
        self.commit()
        self.query("EXPORT (SELECT * FROM vs1.g) INTO JDBC at 'jdbc:exa:{host_port}' user 'sys' identified by 'exasol' TABLE native.g_copy".format(host_port = udf.opts.server))
        self.assertRowsEqualIgnoreOrder(
            self.query("SELECT * FROM native.g"),
            self.query("SELECT * FROM native.g_copy"))
            
    def testExportFromViewWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.g_copy LIKE native.g')
        self.query("CREATE OR REPLACE VIEW native.g_view AS SELECT * FROM vs1.g")
        self.commit()
        self.query("EXPORT native.g_view INTO JDBC at 'jdbc:exa:{host_port}' USER 'sys' IDENTIFIED BY 'exasol' TABLE native.g_copy".format(host_port = udf.opts.server))
        self.assertRowsEqualIgnoreOrder(
            self.query("SELECT * FROM native.g"),
            self.query("SELECT * FROM native.g_copy"))


class MergeAndInsertTest(VSchemaTest):
    def setUp(self):
        self.createNative()
        self.commit()  # commit, otherwise adapter doesn't see tables
        self.createJdbcAdapter(schemaName="ADAPTER", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()

    def testMergeFromQueryWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.t_copy AS SELECT a+1 a, b, c FROM native.t')
        self.query('MERGE INTO native.t_copy t1 USING (SELECT * FROM t) t2 ON (t1.b = t2.b) WHEN MATCHED THEN UPDATE SET a = t2.a')
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT * FROM native.t'),
            self.query('SELECT * FROM native.t_copy'))

    def testMergeFromViewOnVTable(self):
        self.query('CREATE OR REPLACE TABLE native.t_copy AS SELECT a+1 a, b, c FROM native.t')
        self.query('CREATE VIEW native.t_view AS SELECT * FROM t')
        self.commit()
        self.query('MERGE INTO native.t_copy t1 USING native.t_view t2 ON (t1.b = t2.b) WHEN MATCHED THEN UPDATE SET a = t2.a')
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT * FROM native.t'),
            self.query('SELECT * FROM native.t_copy'))

    def testInsertIntoFromQueryWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.t_copy LIKE native.t')
        self.query('INSERT INTO native.t_copy SELECT * FROM vs1.t')
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT * FROM native.t'),
            self.query('SELECT * FROM native.t_copy'))
            
    def testCreateTableAsWithVTable(self):
        self.query('CREATE OR REPLACE TABLE native.t_copy AS SELECT * FROM vs1.t')
        self.assertRowsEqualIgnoreOrder(
            self.query('SELECT * FROM native.t'),
            self.query('SELECT * FROM native.t_copy'))


class AccessControl(VSchemaTest):

    def setUp(self):
        # make sure that there are no Virtual Schemas
        virtualSchemas = self.query('SELECT SCHEMA_NAME FROM EXA_VIRTUAL_SCHEMAS')
        for schema in virtualSchemas:
            self.query('DROP VIRTUAL SCHEMA IF EXISTS {schema} CASCADE'.format(schema=schema[0]))
        # make sure there are no things assigned to PUBLIC role
        self.assertEquals(0, self.queryScalar("SELECT COUNT(*) FROM EXA_DBA_SYS_PRIVS WHERE GRANTEE='PUBLIC'"))
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.commit()

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid = username, pwd = password)
        return client
        
    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username = username))
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username = username, password = password))
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))

    # Test Sys Privileges

    def testSysPrivsExists(self):
        sysPrivs = self.queryScalar("""
            SELECT COUNT(*) FROM EXA_DBA_SYS_PRIVS WHERE GRANTEE='DBA' AND ADMIN_OPTION=TRUE AND
            PRIVILEGE IN ('CREATE VIRTUAL SCHEMA', 'ALTER ANY VIRTUAL SCHEMA', 'ALTER ANY VIRTUAL SCHEMA REFRESH', 'DROP ANY VIRTUAL SCHEMA')
            """)
        self.assertEqual(4, sysPrivs)

    # Create Adapter Script
    
    def testCreateAdapterNoPrivs(self):
        self.createUser("user2", "user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for creating an adapter script'):
            conn.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.FAST_ADAPTER_2 AS
                def adapter_call(request):
                    pass
                /
                '''))
                
    def testCreateAdapterOwnerNoPrivs(self):
        self.createUser("user2", "user2")
        self.query('GRANT CREATE SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        # Even if you are owner you need explicit privs for creating scripts
        conn.query('CREATE SCHEMA ADAPTER_2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for creating an adapter script'):
            conn.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER_2.FAST_ADAPTER_2 AS
                def adapter_call(request):
                    pass
                /
                '''))
                
    def testCreateAdapterWithSysPrivCreateAnyScript(self):
        self.createUser("user2", "user2")
        self.query('GRANT CREATE ANY SCRIPT TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.FAST_ADAPTER_2 AS
            def adapter_call(request):
                pass
            /
            '''))
        self.assertRowsEqual([(1,)], conn.query("SELECT COUNT(*) FROM EXA_ALL_SCRIPTS WHERE SCRIPT_SCHEMA='ADAPTER' AND SCRIPT_NAME='FAST_ADAPTER_2'"))
        
    def testCreateAdapterWithSysPrivCreateScript(self):
        self.createUser("user2", "user2")
        self.query('GRANT CREATE SCRIPT TO user2')
        self.query('GRANT CREATE SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        # Creating Adapter Scripts in non-owned schemas is not allowed
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for creating an adapter script'):
            conn.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.FAST_ADAPTER_2 AS
                def adapter_call(request):
                    pass
                /
                '''))
        self.assertRowsEqual([(0,)], conn.query("SELECT COUNT(*) FROM EXA_ALL_SCRIPTS WHERE SCRIPT_SCHEMA='ADAPTER' AND SCRIPT_NAME='FAST_ADAPTER_2'"))
        # Creating Adapter Scripts in your own schemas is allowed
        conn.query('CREATE SCHEMA ADAPTER_2')
        conn.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER_2.FAST_ADAPTER_2 AS
            def adapter_call(request):
                pass
            /
            '''))
        self.assertRowsEqual([(1,)], conn.query("SELECT COUNT(*) FROM EXA_ALL_SCRIPTS WHERE SCRIPT_SCHEMA='ADAPTER_2' AND SCRIPT_NAME='FAST_ADAPTER_2'"))
        
    # Create Or Replace Adapter Script
        
    def testCreateOrReplaceAdapter(self):
        self.createUser("user2", "user2")
        self.query('GRANT CREATE ANY SCRIPT TO user2')
        self.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.FAST_ADAPTER_REPLACE AS
            def adapter_call(request):
                pass
            /
            '''))
        self.commit()
        oldAdapterObjectId = self.queryScalar("SELECT SCRIPT_OBJECT_ID FROM EXA_ALL_SCRIPTS WHERE SCRIPT_SCHEMA='ADAPTER' AND SCRIPT_NAME='FAST_ADAPTER_REPLACE'")
        # We need sys priv DROP ANY SCRIPT for the old Adapter Script if we want to replace it
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for replacing script'):
            conn.query(udf.fixindent('''
                CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.FAST_ADAPTER_REPLACE AS
                def adapter_call(request):
                    pass
                /
                '''))
        self.assertRowsEqual([(oldAdapterObjectId,)], conn.query("SELECT SCRIPT_OBJECT_ID FROM EXA_ALL_SCRIPTS WHERE SCRIPT_SCHEMA='ADAPTER' AND SCRIPT_NAME='FAST_ADAPTER_REPLACE'"))   # todo replace by conn.queryScalar if available
        # now again with DROP privileges
        self.query('GRANT DROP ANY SCRIPT TO user2')
        self.commit()
        conn.commit()  # Commit, to get new privileges
        conn.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER.FAST_ADAPTER_REPLACE AS
            def adapter_call(request):
                pass
            /
            '''))
        newObjectId = conn.query("SELECT SCRIPT_OBJECT_ID FROM EXA_ALL_SCRIPTS WHERE SCRIPT_SCHEMA='ADAPTER' AND SCRIPT_NAME='FAST_ADAPTER_REPLACE'")   # todo replace by conn.queryScalar if available
        self.assertEqual(1, len(newObjectId))
        self.assertEqual(1, len(newObjectId[0]))
        self.assertNotEqual(oldAdapterObjectId, newObjectId[0][0])

    # Drop Adapter Script
    
    def testDropAdapterNoPrivs(self):
        self.createUser("user2", "user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for dropping script'):
            conn.query('DROP ADAPTER SCRIPT ADAPTER.FAST_ADAPTER')
                
    def testDropAdapterWithSysPriv(self):
        self.createUser("user2", "user2")
        self.query('GRANT DROP ANY SCRIPT TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('DROP ADAPTER SCRIPT ADAPTER.FAST_ADAPTER')

    def testDropAdapterAsSchemaOwner(self):
        self.createUser("user2", "user2")
        self.query('GRANT CREATE SCHEMA TO user2')
        self.query('GRANT CREATE ANY SCRIPT TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE SCHEMA ADAPTER_2')
        conn.query(udf.fixindent('''
            CREATE OR REPLACE PYTHON ADAPTER SCRIPT ADAPTER_2.FAST_ADAPTER_DROP AS
            def adapter_call(request):
                pass
            /
            '''))
        conn.query('DROP ADAPTER SCRIPT ADAPTER_2.FAST_ADAPTER_DROP')

    # Create Virtual Schema

    def testCreateVSchemaNoPrivs(self):
        self.createUser("user2", "user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for creating virtual schema'):
            conn.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
            
    def testCreateVSchemaNoScriptPrivs(self):
        self.createUser("user2", "user2")
        self.query('GRANT DROP ANY VIRTUAL SCHEMA TO user2')
        self.query('GRANT CREATE VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for calling adapter script'):
            conn.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')

    def testCreateVSchema(self):
        self.createUser("user2", "user2")
        self.query('GRANT DROP ANY VIRTUAL SCHEMA TO user2')
        self.query('GRANT CREATE VIRTUAL SCHEMA TO user2')
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')

    # Drop Virtual Schema

    def testDropVSchemaNoDropPrivs(self):
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.createUser("user2", "user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for dropping virtual schema'):
            conn.query('DROP VIRTUAL SCHEMA VS1 CASCADE')
        
    def testDropVSchemaAsOwner(self):
        # user has no privileges to drop, but is owner
        self.createUser("user2", "user2")
        self.query('GRANT CREATE VIRTUAL SCHEMA TO user2')
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        conn.query('DROP VIRTUAL SCHEMA VS1 CASCADE')
        
    def testDropVSchemaWithPrivs(self):
        # user is not owner, but has privileges to drop
        self.createUser("user2", "user2")
        self.query('GRANT CREATE VIRTUAL SCHEMA TO user2')
        self.query('GRANT DROP ANY VIRTUAL SCHEMA TO user2')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('DROP VIRTUAL SCHEMA VS1 CASCADE')

    # Alter Virtual Schema Refresh

    def testAlterVSchemaRefreshNoPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for altering virtual schema'):
            conn.query('ALTER VIRTUAL SCHEMA VS1 REFRESH')

    def testAlterVSchemaRefreshNoScriptPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT ALTER ANY VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        #with self.assertRaisesRegexp(Exception, 'insufficient privileges for calling adapter script'):
        conn.query('ALTER VIRTUAL SCHEMA VS1 REFRESH')

    def testAlterVSchemaRefreshWithAlterSysPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT ALTER ANY VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('ALTER VIRTUAL SCHEMA VS1 REFRESH')
        
    def testAlterVSchemaRefreshWithRefreshSysPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT ALTER ANY VIRTUAL SCHEMA REFRESH TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('ALTER VIRTUAL SCHEMA VS1 REFRESH')
        
    def testAlterVSchemaRefreshWithAlterObjPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT ALTER ON VS1 TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('ALTER VIRTUAL SCHEMA VS1 REFRESH')
        
    def testAlterVSchemaRefreshWithRefreshObjPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT REFRESH ON VS1 TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('ALTER VIRTUAL SCHEMA VS1 REFRESH')
        
    def testAlterVSchemaRefreshAsOwner(self):
        self.createUser("user2", "user2")
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT CREATE VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        conn.query('ALTER VIRTUAL SCHEMA VS1 REFRESH')

    # Alter Virtual Schema Set

    def testAlterVSchemaSetNoPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for altering virtual schema'):
            conn.query("ALTER VIRTUAL SCHEMA VS1 SET FOO='BAR'")
        self.assertEqual(0, self.queryScalar("SELECT COUNT(*) FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'"))
        
    def testAlterVSchemaSetNoScriptPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT ALTER ANY VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query("ALTER VIRTUAL SCHEMA VS1 SET FOO='BAR'")
        self.assertRowsEqual(
            [("FOO","BAR")],
            conn.query("SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'"))
        # with self.assertRaisesRegexp(Exception, 'insufficient privileges for calling adapter script'):
        #     conn.query("ALTER VIRTUAL SCHEMA VS1 SET FOO='BAR'")
        # self.assertEqual(0, self.queryScalar("SELECT COUNT(*) FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'"))
        
    def testAlterVSchemaSetWithAlterSysPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT ALTER ANY VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query("ALTER VIRTUAL SCHEMA VS1 SET FOO='BAR'")
        self.assertRowsEqual(
            [("FOO","BAR")],
            conn.query("SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'"))
            
    def testAlterVSchemaSetWithAlterObjPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT ALTER ON VS1 TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query("ALTER VIRTUAL SCHEMA VS1 SET FOO='BAR'")
        self.assertRowsEqual(
            [("FOO","BAR")],
            conn.query("SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'"))
            
    def testAlterVSchemaSetAsOwner(self):
        self.createUser("user2", "user2")
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT CREATE VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        conn.query("ALTER VIRTUAL SCHEMA VS1 SET FOO='BAR'")
        self.assertRowsEqual(
            [("FOO","BAR")],
            conn.query("SELECT PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES WHERE SCHEMA_NAME = 'VS1'"))

    # Alter Virtual Schema Change Owner

    def testAlterVSchemaChangeOwnerNoPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for altering virtual schema'):
            conn.query("ALTER VIRTUAL SCHEMA VS1 CHANGE OWNER sys")
        self.assertEqual("SYS", self.queryScalar("SELECT SCHEMA_OWNER FROM EXA_VIRTUAL_SCHEMAS WHERE SCHEMA_NAME = 'VS1'"))
        ## added, so that the schema can be dropped -- SPOT-4245
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
                
    def testAlterVSchemaChangeOwnerWithSysPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT ALTER ANY VIRTUAL SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query("ALTER VIRTUAL SCHEMA VS1 CHANGE OWNER USER2")
        self.assertRowsEqual([("USER2",)], conn.query("SELECT SCHEMA_OWNER FROM EXA_VIRTUAL_SCHEMAS WHERE SCHEMA_NAME = 'VS1'"))
        ## added, so that the schema can be dropped -- SPOT-4245
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')

        
    def testAlterVSchemaChangeOwnerObjPrivNotSufficient(self):
        # Object privilege ALTER must not be sufficient to change the owner
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT ALTER ON VS1 TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for altering virtual schema'):
            conn.query("ALTER VIRTUAL SCHEMA VS1 CHANGE OWNER sys")
        self.assertEqual("SYS", self.queryScalar("SELECT SCHEMA_OWNER FROM EXA_VIRTUAL_SCHEMAS WHERE SCHEMA_NAME = 'VS1'"))

    # Open Virtual Schema

    def testOpenVSchema(self):
        # Must be possible without granted privileges
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query("OPEN SCHEMA VS1")
        self.assertEqual("VS1", self.queryScalar("SELECT CURRENT_SCHEMA"))

    # Select Virtual Table

    def testSelectVTableNoPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges: SELECT on table DUMMY'):
            conn.query("SELECT * FROM VS1.DUMMY")
            
    def testSelectVTableNoScriptPrivsWorks(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT SELECT ON VS1 TO USER2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual([('FOO', 'BAR')], conn.query("SELECT * FROM VS1.DUMMY"))
        #with self.assertRaisesRegexp(Exception, 'insufficient privileges for calling adapter script'):
        #    conn.query("SELECT * FROM VS1.DUMMY")
            
    def testSelectVTableWithSysPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT SELECT ANY TABLE TO USER2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual([('FOO', 'BAR')], conn.query("SELECT * FROM VS1.DUMMY"))
        
    def testSelectVTableWithObjPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT SELECT ON VS1 TO USER2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual([('FOO', 'BAR')], conn.query("SELECT * FROM VS1.DUMMY"))
        
    def testSelectVTableAsOwner(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.assertRowsEqual([('FOO', 'BAR')], self.query("SELECT * FROM VS1.DUMMY"))

    # Describe Virtual Table
    
    def testDescribeVTableNoPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges for describing object'):
            conn.query("DESCRIBE VS1.DUMMY")

    def testDescribeVTableWithSysPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT SELECT ANY TABLE TO USER2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('KEY', 'VARCHAR(2000000) UTF8', None, None),('VALUE', 'VARCHAR(2000000) UTF8', None, None)],
            conn.query("DESCRIBE VS1.DUMMY"))

    def testDescribeVTableWithObjPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT SELECT ON VS1 TO USER2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('KEY', 'VARCHAR(2000000) UTF8', None, None),('VALUE', 'VARCHAR(2000000) UTF8', None, None)],
            conn.query("DESCRIBE VS1.DUMMY"))
            
    def testDescribeVTableTableObjPrivsNotSupported(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        with self.assertRaisesRegexp(Exception, 'object privileges for virtual tables are not supported'):
            self.query('GRANT SELECT ON VS1.DUMMY TO USER2')

    def testDescribeVTableAsOwner(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.assertRowsEqual(
            [('KEY', 'VARCHAR(2000000) UTF8', None, None),('VALUE', 'VARCHAR(2000000) UTF8', None, None)],
            self.query("DESCRIBE VS1.DUMMY"))

    # Create Views with Virtual Tables

    def testCreateViewWithVTableNoSelectPrivs(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT CREATE VIEW TO user2')
        self.query('GRANT CREATE SCHEMA TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE SCHEMA VIEWS')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges: SELECT on table DUMMY'):
            conn.query('CREATE VIEW VIEWS.VTABLE_VIEW AS SELECT * FROM VS1.DUMMY')
            
    def testCreateViewWithVTableWithSelectPriv(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT CREATE VIEW TO user2')
        self.query('GRANT CREATE SCHEMA TO user2')
        self.query('GRANT SELECT ON VS1 TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE SCHEMA VIEWS')
        conn.query('CREATE VIEW VIEWS.VTABLE_VIEW AS SELECT * FROM VS1.DUMMY')
        self.assertRowsEqual(
            self.query('SELECT * FROM VS1.DUMMY'),
            conn.query('SELECT * FROM VIEWS.VTABLE_VIEW'))

    # Query Views with Virtual Tables
    def testQueryViewWithVTable(self):
        self.createUser("user2", "user2")
        self.query('CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')
        #self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')
        self.query('GRANT CREATE VIEW TO user2')
        self.query('GRANT CREATE SCHEMA TO user2')
        self.query('GRANT SELECT ON VS1 TO user2')
        self.commit()
        conn = self.getConnection('user2', 'user2')
        conn.query('CREATE SCHEMA VIEWS')
        conn.query('CREATE VIEW VIEWS.VTABLE_VIEW AS SELECT * FROM VS1.DUMMY')
        conn.commit()
        # The owner of the view (user2) has privileges, so works
        self.assertRowsEqual(
            self.query('SELECT * FROM VS1.DUMMY'),
            self.query('SELECT * FROM VIEWS.VTABLE_VIEW'))
        self.query('REVOKE SELECT ON VS1 FROM user2')
        self.commit()
        # The owner of the view has no more privileges to SELECT from vtable
        with self.assertRaisesRegexp(Exception, 'insufficient privileges: SELECT on table DUMMY'):
            self.query('SELECT * FROM VIEWS.VTABLE_VIEW')

      
    def testRowlevelSecurityUseCase(self):
        self.query('''CREATE SCHEMA adapter_4091_schema''')
        self.query('''CREATE SCHEMA data_4091_schema''')
        self.query('''CREATE TABLE data_4091_schema.t(a1 varchar(100), a2 varchar(100), userName varchar(100))''')
        self.query('''INSERT INTO data_4091_schema.t values('a', 'b', 'SYS')''')
        self.query('''INSERT INTO data_4091_schema.t values('c', 'd', 'SYS')''')
        self.query('''INSERT INTO data_4091_schema.t values('e', 'f', 'U2')''')
        self.query('''INSERT INTO data_4091_schema.t values('g', 'h', 'U2')''')
        self.query('''INSERT INTO data_4091_schema.t values('i', 'j', 'USER4091')''')
        self.query('''INSERT INTO data_4091_schema.t values('k', 'l', 'USER4091')''')

        all_data_rows = [('a','b','SYS'),('c','d','SYS'),
                         ('e','f','U2'),('g','h','U2'),
                         ('i','j','USER4091'),('k','l','USER4091')]
        user4091_data_rows = [('i','j','USER4091'),('k','l','USER4091')]        
        self.query('''
CREATE OR REPLACE PYTHON ADAPTER SCRIPT adapter_4091_schema.rls_adapter AS
import json
import string
def adapter_call(request):
    # database expects utf-8 encoded string of type str. unicode not yet supported
    root = json.loads(request)
    if root["type"] == "createVirtualSchema":
        res = {
            "type": "createVirtualSchema",
            "schemaMetadata": {
                "tables": [
                {
                    "name": "T",
                    "columns": [{
                        "name": "a1",
                        "dataType": {"type": "VARCHAR", "size": 2000000}
                    },{
                        "name": "a2",
                        "dataType": {"type": "VARCHAR", "size": 2000000}
                    },{
                        "name": "userName",
                        "dataType": {"type": "VARCHAR", "size": 100}
                    }]
                }]
            }
        }
        return json.dumps(res).encode('utf-8')
    elif root["type"] == "dropVirtualSchema":
        return json.dumps({"type": "dropVirtualSchema"}).encode('utf-8')
    elif root["type"] == "setProperties":
        return json.dumps({"type": "setProperties"}).encode('utf-8')
    elif root["type"] == "refresh":
        return json.dumps({"type": "refresh"}).encode('utf-8')
    if root["type"] == "getCapabilities":
        return json.dumps({
            "type": "getCapabilities",
            "capabilities": []
            }).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
    elif root["type"] == "pushdown":
        res = {
            "type": "pushdown",
            "sql": "SELECT * FROM data_4091_schema.t WHERE userName = current_user or current_user = 'SYS'"
        }
        return json.dumps(res).encode('utf-8')
    else:
        raise ValueError('Unsupported callback')
/
''')
        #self.query('''DROP VIRTUAL SCHEMA RSL_SCHEMA CASCADE''')
        self.query('''CREATE VIRTUAL SCHEMA RSL_SCHEMA USING adapter_4091_schema.rls_adapter''')
        self.assertRowsEqualIgnoreOrder(all_data_rows,self.query('''SELECT * FROM RSL_SCHEMA.T'''))
        self.query('''CREATE SCHEMA USER_SCHEMA''')
        self.query('''CREATE OR REPLACE VIEW USER_SCHEMA.RSL_VIEW AS SELECT * FROM RSL_SCHEMA.T''')
        self.assertRowsEqualIgnoreOrder(all_data_rows,self.query('''SELECT * FROM USER_SCHEMA.RSL_VIEW'''))
        self.createUser("user4091","user4091")
        self.query('''GRANT CREATE SESSION TO user4091''')
        self.query('''GRANT SELECT ON USER_SCHEMA.RSL_VIEW TO user4091''')
        #GRANT EXECUTE ON adapter_4091_schema.fast_adapter to U1;
        self.commit()
        conn = self.getConnection('user4091','user4091')
        self.assertRowsEqualIgnoreOrder(user4091_data_rows, conn.query('''SELECT * FROM USER_SCHEMA.RSL_VIEW'''))
        self.query('''DROP USER USER4091''')
        self.query('''DROP VIRTUAL SCHEMA RSL_SCHEMA CASCADE''')
        self.query('''DROP SCHEMA USER_SCHEMA CASCADE''')
        self.query('''DROP ADAPTER SCRIPT adapter_4091_schema.rls_adapter''')
        self.query('''DROP SCHEMA data_4091_schema cascade''')
        self.query('''DROP SCHEMA adapter_4091_schema cascade''')

            
    # EXA_*_VIRTUAL_SCHEMA_PROPERTIES
    
    # Test Variants:
    #-- owner | obj priv ALTER schema | sys ALTER ANY VS
    #-- no    | no                    | no
    #-- yes   | no                    | no
    #-- no    | direct                | no
    #-- no    | via role              | no
    #-- no    | via public role       | no
    #-- no    | no                    | direct
    #-- no    | no                    | via role
    #-- no    | no                    | via public role
    # - Other privs not sufficient (ALTER/DROP ANY VIRTUAL SCHEMA REFRESH, SELECT ANY TABLE, SELECT on schema, REFRESH on schema)
    
    def testSysTableVSchemaPropertiesNoDBAAccess(self):
        self.createUser("user2", "user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        with self.assertRaisesRegexp(Exception, 'insufficient privileges: SELECT on table EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES'):
            conn.query("SELECT * FROM EXA_DBA_VIRTUAL_SCHEMA_PROPERTIES")

    def testSysTableVSchemaPropertiesNoPrivs(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))

    # User has privs, but not the ones required to view virtual schema properties
    def testSysTableVSchemaPropertiesWrongPrivs(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        #self.query("GRANT SELECT ANY TABLE to user2")
        #self.query("GRANT ALTER ANY SCHEMA to user2")
        #self.query("GRANT ALTER ANY VIRTUAL SCHEMA REFRESH to user2")
        #self.query("GRANT SELECT on VS1 to user2")
        #self.query("GRANT REFRESH on VS1 to user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
            
    # def testSysTableVSchemaPropertiesOwner(self):
    #     self.createUser("user2", "user2")
    #     self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
    #     self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
    #     self.query("ALTER VIRTUAL SCHEMA VS1 CHANGE OWNER user2")
    #     self.commit()
    #     conn = self.getConnection('user2', 'user2')
    #     self.assertRowsEqual(
    #         [('VS1', 'P', 'V1')],
    #         conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES"))
    #     self.assertRowsEqual(
    #         [('VS1', 'P', 'V1')],
    #         conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
    #      ## added, so that the schema can be dropped -- SPOT-4245
    #     self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2') 

    def testSysTableVSchemaPropertiesAlterObjPriv(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        self.query("GRANT ALTER ON VS1 TO user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('VS1', 'P', 'V1')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))


    def testSysTableVSchemaPropertiesAlterObjPrivViaRole(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        self.query("DROP ROLE IF EXISTS role_01")
        self.query("DROP ROLE IF EXISTS role_02")
        self.query("DROP ROLE IF EXISTS role_03")
        self.query("CREATE ROLE role_01")
        self.query("CREATE ROLE role_02")
        self.query("CREATE ROLE role_03")
        self.query("GRANT role_01 TO role_02")
        self.query("GRANT role_02 TO role_03")
        self.query("GRANT role_03 TO user2")
        self.query("GRANT ALTER ON VS1 TO role_01")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('VS1', 'P', 'V1')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
            
    def testSysTableVSchemaPropertiesAlterObjPrivViaPublicRole(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        self.query("GRANT ALTER ON VS1 TO public")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('VS1', 'P', 'V1')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
            
    def testSysTableVSchemaPropertiesAlterSysPriv(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        self.query("GRANT ALTER ANY VIRTUAL SCHEMA TO user2")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('VS1', 'P', 'V1'), ('VS2', 'P', 'V2')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES ORDER BY SCHEMA_NAME"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
            
    def testSysTableVSchemaPropertiesAlterSysPrivViaRole(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        self.query("DROP ROLE IF EXISTS role_01")
        self.query("DROP ROLE IF EXISTS role_02")
        self.query("DROP ROLE IF EXISTS role_03")
        self.query("CREATE ROLE role_01")
        self.query("CREATE ROLE role_02")
        self.query("CREATE ROLE role_03")
        self.query("GRANT role_01 TO role_02")
        self.query("GRANT role_02 TO role_03")
        self.query("GRANT role_03 TO user2")
        self.query("GRANT ALTER ANY VIRTUAL SCHEMA TO role_01")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('VS1', 'P', 'V1'), ('VS2', 'P', 'V2')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES ORDER BY SCHEMA_NAME"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
            
    def testSysTableVSchemaPropertiesAlterSysPrivViaPublicRole(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        self.query("GRANT ALTER ANY VIRTUAL SCHEMA TO public")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('VS1', 'P', 'V1'), ('VS2', 'P', 'V2')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES ORDER BY SCHEMA_NAME"))
        self.assertRowsEqual(
            [],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
        self.query("REVOKE ALTER ANY VIRTUAL SCHEMA FROM public")
        self.commit()

    def testSysTableVSchemaPropertiesMultiPrivs(self):
        self.createUser("user2", "user2")
        self.query("CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER WITH P='V1'")
        self.query("CREATE VIRTUAL SCHEMA VS2 USING ADAPTER.FAST_ADAPTER WITH P='V2'")
        # Give the user all kinds of privs
        self.query("GRANT ALTER ANY VIRTUAL SCHEMA TO user2")
        self.query("GRANT ALTER on VS1 TO user2")
        self.query("ALTER VIRTUAL SCHEMA VS1 CHANGE OWNER user2")
        self.query("DROP ROLE IF EXISTS role_01")
        self.query("DROP ROLE IF EXISTS role_02")
        self.query("DROP ROLE IF EXISTS role_03")
        self.query("CREATE ROLE role_01")
        self.query("CREATE ROLE role_02")
        self.query("CREATE ROLE role_03")
        self.query("GRANT role_01 TO role_02")
        self.query("GRANT role_02 TO role_03")
        self.query("GRANT role_03 TO user2")
        self.query("GRANT ALTER ANY VIRTUAL SCHEMA TO role_01")
        self.query("GRANT ALTER on VS1 TO role_01")
        self.commit()
        conn = self.getConnection('user2', 'user2')
        self.assertRowsEqual(
            [('VS1', 'P', 'V1'), ('VS2', 'P', 'V2')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_ALL_VIRTUAL_SCHEMA_PROPERTIES ORDER BY SCHEMA_NAME"))
        self.assertRowsEqual(
            [('VS1', 'P', 'V1')],
            conn.query("SELECT SCHEMA_NAME, PROPERTY_NAME, PROPERTY_VALUE FROM EXA_USER_VIRTUAL_SCHEMA_PROPERTIES"))
        ## added, so that the schema can be dropped -- SPOT-4245
        self.query('GRANT EXECUTE ON ADAPTER.FAST_ADAPTER TO user2')

class MiscTest(VSchemaTest):

    def testBig(self):
        self.query('DROP SCHEMA IF EXISTS NBIG CASCADE')
        self.query('CREATE SCHEMA NBIG')
        size = 1
        sizeColumns = 1000
        for i in range(0,size):
            columns = ""
            for j in range(0,sizeColumns):
                if (j != 0):
                    columns += ", "
                columns += "a{col} int, b{col} varchar(100)".format(col=j)
            queryString = "CREATE TABLE T{idx}(".format(idx=i)
            queryString += columns + ")"
            self.query(queryString)
        self.commit()  # commit, otherwise adapter doesn't see tables
        for i in range(0,size):
            values = ""
            for j in range(0,sizeColumns):
                if (j != 0):
                    values += ", "
                values += "{col}, '{col}'".format(col=j)
            queryString = "INSERT INTO T{idx} VALUES (".format(idx=i)
            queryString += values + ")"
            self.query(queryString)
        self.createJdbcAdapter()
        self.createVirtualSchemaJdbc("VS1", "NBIG", "ADAPTER.JDBC_ADAPTER", True)
        select = "SELECT "
        for i in range(0,size):
            if (i != 0):
                select += ", "
            for j in range(0,sizeColumns):
                if (j != 0):
                    select += ", "
                select += "T{idx}.a{col} a{idx}_{col}, T{idx}.b{col} b{idx}_{col} ".format(idx=i, col=j);
        select += "FROM "
        selectNative = select
        for i in range(0,size):
            if (i != 0):
                select += ", "
            select += "VS1.T{idx}".format(idx=i)
        for i in range(0,size):
            if (i != 0):
                selectNative += ", "
            selectNative += "NBIG.T{idx}".format(idx=i)
        rowsNative = self.query(selectNative)
        rows = self.query(select)
        self.assertRowsEqual(rows, rowsNative)
        self.query('DROP SCHEMA IF EXISTS NBIG CASCADE')


    def testLargeData(self):
        self.query('DROP SCHEMA IF EXISTS NBIG2 CASCADE')
        self.query('CREATE SCHEMA NBIG2')
        self.query('CREATE TABLE Tlarge(a varchar(2000000)) ')
        self.commit()
        size = 10
        for i in range(0,size):

            value =""
            char = i % 10
            c = `i`
            for j in range(0,2000000):
                value+= c
            queryString = "INSERT INTO Tlarge VALUES ('" + value +"')"
            self.query(queryString)
        self.createJdbcAdapter()
        self.createVirtualSchemaJdbc("VS1", "NBIG2", "ADAPTER.JDBC_ADAPTER", True)
        rowsNative = self.query("SELECT * FROM NBIG2.Tlarge");
        rows = self.query("SELECT * FROM VS1.Tlarge");
        self.assertRowsEqual(rows, rowsNative)
        self.query('DROP SCHEMA IF EXISTS NBIG2 CASCADE')

    def testLargeAdapterNotes(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_NOTES CASCADE')
        self.createNotesAdapter(schemaName="ADAPTER", adapterName="NOTES_ADAPTER")
        self.query('CREATE  VIRTUAL SCHEMA VS_NOTES USING ADAPTER.NOTES_ADAPTER')
        self.query('SELECT * FROM VS_NOTES.DUMMY')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_NOTES CASCADE')


    def testManyAdapterNoteEntries(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_NOTES2 CASCADE')
        self.createSecondNotesAdapter(schemaName="ADAPTER", adapterName="SECOND_NOTES_ADAPTER")
        self.query('CREATE  VIRTUAL SCHEMA VS_NOTES2 USING ADAPTER.SECOND_NOTES_ADAPTER')
        self.query('SELECT * FROM VS_NOTES2.DUMMY')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_NOTES2 CASCADE')


    def createNotesAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER"):
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
                    c = "0"
                    value = ""
                    value += '\"'
                    for j in range(0,60000000):
                        value+= c
                    value += '\"'
                    res = {{
                        "type": "createVirtualSchema",
                        "schemaMetadata": {{
                            "adapterNotes": "",
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
                    res["schemaMetadata"]["adapterNotes"] = value
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

    def createSecondNotesAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER"):
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
                            "adapterNotes": "",
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
                    tmp = {{}}
                    for j in range(0,1000000):
                        key = "k"+`j`
                        tmp[key] = `j`
                    res["schemaMetadata"]["adapterNotes"] = '\"' + json.dumps(tmp) + '\"'
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

class ExplainVirtualBasic(VSchemaTest):

    def setUp(self):
        self.query('DROP FORCE VIRTUAL SCHEMA IF EXISTS VS1 CASCADE')
        self.createFastAdapter(schemaName="ADAPTER", adapterName="FAST_ADAPTER")
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.FAST_ADAPTER", True)
        self.commit()

    def testExplainWithoutVirtual(self):
        with self.assertRaisesRegexp(Exception, 'syntax error, unexpected SELECT_'):
            self.query('EXPLAIN SELECT * FROM DUMMY')

    def testExplainWithoutVirtualTable(self):
        with self.assertRaisesRegexp(Exception, 'Explain virtual not possible on queries without virtual tables'):
            self.query('EXPLAIN VIRTUAL (SELECT 2 FROM DUAL);')

    def testExplainVirtualWithSimpleRequest(self):
        rows = self.query('EXPLAIN VIRTUAL SELECT * FROM DUMMY;')
        self.assertEqual([1,], self.getColumn(rows,0))
        self.assertEqual(['''SELECT * FROM (VALUES ('FOO', 'BAR')) t''',], self.getColumn(rows,1))
        self.assertEqual(['DUMMY',], self.getColumn(rows,3))

    def testExplainVirtualWithSubqueryId(self):
        rows = self.query('SELECT PUSHDOWN_ID FROM (EXPLAIN VIRTUAL SELECT * FROM DUMMY);')
        self.assertRowsEqual([(1,)], rows)

    def testExplainVirtualWithSubquerySQL(self):
        rows = self.query('SELECT PUSHDOWN_SQL FROM (EXPLAIN VIRTUAL SELECT * FROM DUMMY);')
        self.assertRowsEqual([('''SELECT * FROM (VALUES ('FOO', 'BAR')) t''',)], rows)

    def testExplainVirtualWithSubqueryJson(self):
        rows = self.query('SELECT PUSHDOWN_JSON FROM (EXPLAIN VIRTUAL SELECT * FROM DUMMY);')
        self.assertTrue("schemaMetadataInfo" in rows[0][0])

    def testExplainVirtualWithSubqueryInvolvedTables(self):
        rows = self.query('SELECT PUSHDOWN_INVOLVED_TABLES FROM (EXPLAIN VIRTUAL SELECT * FROM DUMMY);')
        self.assertRowsEqual([('''DUMMY''',)], rows)

    def testExplainVirtualWithSubqueryALL(self):
        rows = self.query('SELECT * FROM (EXPLAIN VIRTUAL SELECT * FROM DUMMY);')
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(rows[0]), 4)
        self.assertEqual(rows[0][0], 1)
        self.assertEqual(rows[0][1], '''SELECT * FROM (VALUES ('FOO', 'BAR')) t''')
        self.assertEqual(rows[0][3], '''DUMMY''')
        self.assertTrue("schemaMetadataInfo" in rows[0][2])
        self.assertTrue("capabilities" in rows[0][2])
        self.assertTrue("pushdownRequest" in rows[0][2])
        self.assertTrue("pushdown" in rows[0][2])

    def testExplainVirtualWithCreate(self):
        with self.assertRaisesRegexp(Exception, 'syntax error, unexpected CREATE_'):
            self.query('EXPLAIN VIRTUAL CREATE VIRTUAL SCHEMA VS1 USING ADAPTER.FAST_ADAPTER')

    def testExplainVirtualWithAlter(self):
        with self.assertRaisesRegexp(Exception, 'syntax error, unexpected ALTER_'):
            self.query('''EXPLAIN VIRTUAL ALTER VIRTUAL SCHEMA VS1 SET UNUSED='FOO' ''')

    def testExplainVirtualWithAlterRefresh(self):
        with self.assertRaisesRegexp(Exception, 'syntax error, unexpected ALTER_'):
            self.query('EXPLAIN VIRTUAL ALTER VIRTUAL SCHEMA VS1 REFRESH')

    def testExplainVirtualWithDrop(self):
        with self.assertRaisesRegexp(Exception, 'syntax error, unexpected DROP_'):
            self.query('EXPLAIN VIRTUAL DROP VIRTUAL SCHEMA VS1')

    def testExplainVirtualWithInvalidPushdownSql(self):
        self.query(udf.fixindent('''
        CREATE OR REPLACE PYTHON ADAPTER SCRIPT adapter.invalid_fast_adapter AS
        import json
        import string
        def adapter_call(request):
            # database expects utf-8 encoded string of type str. unicode not yet supported
            root = json.loads(request)
            if root["type"] == "createVirtualSchema":
                res = {
                    "type": "createVirtualSchema",
                    "schemaMetadata": {
                        "tables": [
                        {
                            "name": "DUMMY",
                            "columns": [{
                                "name": "A",
                                "dataType": {"type": "VARCHAR", "size": 2000000}
                            },{
                                "name": "B",
                                "dataType": {"type": "VARCHAR", "size": 2000000}
                            }]
                        }]
                    }
                }
                return json.dumps(res).encode('utf-8')
            elif root["type"] == "dropVirtualSchema":
                return json.dumps({"type": "dropVirtualSchema"}).encode('utf-8')
            if root["type"] == "getCapabilities":
                return json.dumps({
                    "type": "getCapabilities",
                    "capabilities": []
                    }).encode('utf-8') # database expects utf-8 encoded string of type str. unicode not yet supported.
            elif root["type"] == "pushdown":
                res = {
                    "type": "pushdown",
                    "sql": "IMPORT FROM JDBC AT 'jdbc:exa:non-existing-host:1234' USER 'alice' IDENTIFIED BY 'bob' STATEMENT 'SELECT * FROM non-existing-table'"
                }
                return json.dumps(res).encode('utf-8')
            else:
                raise ValueError('Unsupported callback')
        /
            '''))
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_INVALID_EXPLAIN CASCADE')
        self.query('CREATE VIRTUAL SCHEMA VS_INVALID_EXPLAIN USING ADAPTER.invalid_fast_adapter')
        rows = self.query('SELECT PUSHDOWN_SQL FROM (EXPLAIN VIRTUAL SELECT * FROM VS_INVALID_EXPLAIN.dummy)')
        self.assertRowsEqual([("""IMPORT FROM JDBC AT 'jdbc:exa:non-existing-host:1234' USER 'alice' IDENTIFIED BY 'bob' STATEMENT 'SELECT * FROM non-existing-table'""",)], rows)

        self.commit()

class ExplainVirtualPushdown(VSchemaTest):
    setupDone = False

    def setUp(self):
        # TODO This is another ugly workaround for the problem that the framework doesn't offer us a query in classmethod setUpClass. Rewrite!
        if self.__class__.setupDone:
            self.query(''' CLOSE SCHEMA ''')
            return

        self.createJdbcAdapter()
        self.createNative()
        self.commit()  # We have to commit, otherwise the adapter won't see these tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        self.commit()
        self.query(''' CLOSE SCHEMA ''')
        self.__class__.setupDone = True

    def testPushdownResponses(self):
        # Single Group
        self.compareWithExplainVirtual('''
            SELECT a, c FROM {v}.T;
        ''', '''SELECT A, C FROM NATIVE.T''')
        self.compareWithExplainVirtual('''
            SELECT t1.a FROM {v}.t t1, {v}.t t2
        ''', ['''SELECT A FROM NATIVE.T''','''SELECT true FROM NATIVE.T'''])
        self.compareWithExplainVirtual('''
            SELECT (a+1) a1 FROM {v}.t
        ''', '''SELECT (A + 1) FROM NATIVE.T''')

    def testNestedPushdowns(self):
        self.compareWithExplainVirtual('''
            SELECT a FROM (SELECT a FROM VS1.t ORDER BY false);
        ''', '''SELECT A FROM NATIVE.T ORDER BY false''')
        rows = self.query('''
            SELECT PUSHDOWN_SQL||'X' FROM (EXPLAIN VIRTUAL SELECT a FROM (SELECT a FROM VS1.t ORDER BY false));
        ''')
        self.assertRowsEqual([('''SELECT A FROM NATIVE.T ORDER BY falseX''',)], rows)

        self.compareWithExplainVirtual('''
            SELECT a FROM (SELECT a FROM VS1.t ORDER BY false), (SELECT b FROM VS1.t ORDER BY false);
        ''', ['SELECT A FROM NATIVE.T ORDER BY false', 'SELECT NULL FROM NATIVE.T ORDER BY false']) # review!
        rows = self.query('''
            SELECT PUSHDOWN_SQL||'X' FROM (EXPLAIN VIRTUAL SELECT a FROM (SELECT a FROM VS1.t ORDER BY false), (SELECT b FROM VS1.t ORDER BY false));
        ''')
        self.assertRowsEqualIgnoreOrder([('''SELECT A FROM NATIVE.T ORDER BY falseX''',),('''SELECT NULL FROM NATIVE.T ORDER BY falseX''',)], rows)

        self.compareWithExplainVirtual('''
            SELECT * FROM VS1.t WHERE a IN (SELECT DISTINCT a FROM VS1.t ORDER BY a DESC LIMIT 2);
        ''', ['''SELECT * FROM NATIVE.T''','''SELECT A FROM NATIVE.T GROUP BY A ORDER BY A DESC LIMIT 2'''])
        rows = self.query('''
            SELECT PUSHDOWN_SQL||'X' FROM (EXPLAIN VIRTUAL SELECT * FROM VS1.t WHERE a IN (SELECT DISTINCT a FROM VS1.t ORDER BY a DESC LIMIT 2));
        ''')
        self.assertRowsEqualIgnoreOrder([('''SELECT * FROM NATIVE.TX''',),('''SELECT A FROM NATIVE.T GROUP BY A ORDER BY A DESC LIMIT 2X''',)], rows)

        rows = self.query('''
            SELECT * FROM (EXPLAIN VIRTUAL SELECT a FROM VS1.t ORDER BY false), (EXPLAIN VIRTUAL SELECT b FROM VS1.t ORDER BY false);
        ''')
        self.assertEquals(['''SELECT A FROM NATIVE.T ORDER BY false''', '''SELECT A FROM NATIVE.T ORDER BY false'''], self.getColumn(rows, 1))
        self.assertEquals(['''SELECT A FROM NATIVE.T ORDER BY false''', '''SELECT B FROM NATIVE.T ORDER BY false'''], self.getColumn(rows, 5))

    def testJoins(self):
        # Equi Join
        self.compareWithExplainVirtual('''
            select t1.a FROM {v}.t t1 join {v}.t t2 on t1.b=t2.b
        ''', ['''SELECT A, B FROM NATIVE.T''','''SELECT B FROM NATIVE.T'''])
        # Outer Join
        self.compareWithExplainVirtual('''
            select * FROM {v}.t t1 left join {v}.t t2 on t1.a=t2.a where coalesce(t2.a, 1) = 1
        ''', ['''SELECT * FROM NATIVE.T''', '''SELECT * FROM NATIVE.T'''])
        # Cross Join
        self.compareWithExplainVirtual('''
            select t1.a FROM {v}.t t1, {v}.t t2
        ''', ['''SELECT A FROM NATIVE.T''', '''SELECT true FROM NATIVE.T'''])
        # Join with native table
        self.compareWithExplainVirtual('''
            select * from {v}.t vt join {n}.t nt on vt.a = nt.a where nt.a = 1
        ''', '''SELECT * FROM NATIVE.T''')

    def testSelectListExpressions(self):
        self.compareWithExplainVirtual('''
            select a+1 from {v}.t order by c desc
        ''', '''SELECT (A + 1) FROM NATIVE.T ORDER BY C DESC''')

    def testPredicates(self):
        self.compareWithExplainVirtual('''
            SELECT a=1, b FROM {v}.t WHERE a=(a*2/2)
        ''', '''SELECT A = 1, B FROM NATIVE.T WHERE A = ((A * 2) / 2)''')


    def testOrderByLimit(self):
        self.compareWithExplainVirtual('''
            select a+1 as a1, c from {v}.t order by a+1
        ''', '''SELECT (A + 1), C FROM NATIVE.T ORDER BY (A + 1)''')

    def testAggregation(self):
        # Single Group
        self.compareWithExplainVirtual('''
            select count(*) from {v}.t
        ''', '''SELECT COUNT(*) FROM NATIVE.T''')

        # Group By Expression
        self.compareWithExplainVirtual('''
            select a*2, count(*), max(b) from {v}.t group by a*2
        ''', '''SELECT (A * 2), COUNT(*), MAX(B) FROM NATIVE.T GROUP BY (A * 2)''')

        # Aggregation On Join
        self.compareWithExplainVirtual('''
            select sum(t1.a) from {v}.t t1, {v}.t t2 group by t1.a
        ''',  ['SELECT A FROM NATIVE.T', 'SELECT true FROM NATIVE.T'])

    def testScalarFunctions(self):
        # Aggregation On Join
        self.compareWithExplainVirtual('''
            select * from {v}.t where abs(a) = 1
        ''',  'SELECT * FROM NATIVE.T WHERE ABS(A) = 1')

    def testMultiPushdown(self):
        self.createVirtualSchemaJdbc("VS2", "NATIVE", "ADAPTER.JDBC_ADAPTER", True)
        # Create an additional virtual schema using another adapter
        self.createJdbcAdapter(schemaName="ADAPTER2", adapterName="JDBC_ADAPTER")
        self.createVirtualSchemaJdbc("VS3", "NATIVE", "ADAPTER2.JDBC_ADAPTER", True)
        # 1 virtual schema, n virtual tables
        self.compareWithExplainVirtual('''
            select * from {v}.t t1, {v}.t t2, {v}.t t3 where t1.a = t2.a and t2.a = t3.a;
        ''',  ['SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T'])

        # 1 adapter, n virtual schemas
        self.compareWithExplainVirtual('''
            select * from {v}.t t1, {v2}.t t2, {v}.t t3 where t1.a = t2.a and t2.a = t3.a;
        ''',  ['SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T'])

        # different adapters, different schemas
        self.compareWithExplainVirtual('''
            select * from {v}.t t1, {v3}.t t2 where t1.a = t2.a;
        ''', ['SELECT * FROM NATIVE.T', 'SELECT * FROM NATIVE.T'])
        self.compareWithExplainVirtual('''
            select * from {v}.t t1, (select a, b from {v3}.t) t2 where t1.a = t2.a;
        ''',  ['SELECT * FROM NATIVE.T','SELECT A, B FROM NATIVE.T'])
        self.compareWithExplainVirtual('''
            select * from {v}.t where a in (select distinct a from {v3}.t order by a desc limit 2);
        ''',  ['SELECT * FROM NATIVE.T','SELECT A FROM NATIVE.T GROUP BY A ORDER BY A DESC LIMIT 2'])

    def testWithAnalytical(self):
        self.compareWithExplainVirtual('''
            SELECT k, v1, sum(v1) over (PARTITION BY k ORDER BY v1) AS SUM FROM {v}.g order by k desc, sum;
        ''',  'SELECT K, V1 FROM NATIVE.G')

    def testMixed(self):
        # Special Case: c*c is removed from select list, so only lookups in selectlist. Should still pushdown agg.
        self.compareWithExplainVirtual('''
            SELECT count(a) FROM (
              SELECT a,c*c as x, sum(c) mysum FROM {v}.t GROUP BY a,c*c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''',  '''SELECT A FROM NATIVE.T WHERE (C * C) < 15 GROUP BY (C * C), A HAVING 2 < SUM(C)''') # review!

        # ... same with b only in filter
        self.compareWithExplainVirtual('''
            SELECT count(a) FROM (
              SELECT a,c*c as x, sum(c) mysum FROM {v}.t WHERE b!='f' GROUP BY a,c*c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''',  '''SELECT A FROM NATIVE.T WHERE (B != 'f' AND (C * C) < 15) GROUP BY (C * C), A HAVING 2 < SUM(C)''') # review!

        # ... same with join
        self.compareWithExplainVirtual('''
            SELECT count(a) FROM (
              SELECT t1.a,t1.c*t1.c as x, sum(t1.c) mysum FROM {v}.t t1 JOIN {v}.t t2 ON t1.b=t2.b GROUP BY t1.a,t1.c*t1.c) subsel
            WHERE subsel.x<15 AND mysum>2;
        ''',  ['SELECT * FROM NATIVE.T WHERE (C * C) < 15', 'SELECT B FROM NATIVE.T'])

class AdapterNotes(VSchemaTest):

    def testStringWithQuotesSchema(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement="\\\"string\\\"")
        self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('SELECT * FROM VS_ILLEGAL_NOTES.DUMMY')
        rows = self.query("SELECT ADAPTER_NOTES FROM EXA_VIRTUAL_SCHEMAS WHERE SCHEMA_NAME='VS_ILLEGAL_NOTES'")
        self.assertRowsEqual([('string',)], rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonObjectSchema(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement="[{\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"KEY\\\"}, {\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"VALUE\\\"}]")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonNonExistingObjectSchema(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement="[{\\\"data\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"KEY\\\"}, {\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"VALUE\\\"}]")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonBooleanSchema(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement="true")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testIllegalQuoteSchema(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement='''\\\"\\\\\\'t\\\'\\\"''')
        with self.assertRaisesRegexp(Exception, "Bad escape sequence in string."):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testAdapterNoteWithoutQuotesSchema(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement="no quotes")
        with self.assertRaisesRegexp(Exception, "Syntax error: value, object or array expected."):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testIncompleteAdapterNoteSchema(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement="\\\"x")
        with self.assertRaisesRegexp(Exception, "Missing ',' or '}' in object declaration"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testStringWithQuotesTable(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement2="\\\"string\\\"")
        self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('SELECT * FROM VS_ILLEGAL_NOTES.DUMMY')
        rows = self.query("SELECT ADAPTER_NOTES FROM EXA_DBA_VIRTUAL_TABLES WHERE TABLE_SCHEMA='VS_ILLEGAL_NOTES' AND TABLE_NAME='DUMMY'")
        self.assertRowsEqual([('string',)], rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonObjectTable(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement2="[{\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"KEY\\\"}, {\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"VALUE\\\"}]")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonNonExistingObjectTable(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement2="[{\\\"data\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"KEY\\\"}, {\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"VALUE\\\"}]")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonBooleanTable(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement2="true")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testIllegalQuoteTable(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement2='''\\\"\\\\\\'t\\\'\\\"''')
        with self.assertRaisesRegexp(Exception, "Bad escape sequence in string."):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testAdapterNoteWithoutQuotesTable(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement2="no quotes")
        with self.assertRaisesRegexp(Exception, "Syntax error: value, object or array expected."):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testIncompleteAdapterNoteTable(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement2="\\\"x")
        with self.assertRaisesRegexp(Exception, "Missing ',' or '}' in object declaration"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testStringWithQuotesColumn(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement3="\\\"string\\\"")
        self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('SELECT * FROM VS_ILLEGAL_NOTES.DUMMY')
        rows = self.query("SELECT ADAPTER_NOTES FROM EXA_DBA_VIRTUAL_COLUMNS WHERE COLUMN_SCHEMA='VS_ILLEGAL_NOTES' AND COLUMN_TABLE='DUMMY' ORDER BY COLUMN_NAME")
        self.assertRowsEqual([('string',),(None,)], rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonObjectColumn(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement3="[{\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"KEY\\\"}, {\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"VALUE\\\"}]")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonNonExistingObjectColumn(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement3="[{\\\"data\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"KEY\\\"}, {\\\"dataType\\\": {\\\"type\\\": \\\"VARCHAR\\\", \\\"size\\\": 100}, \\\"name\\\": \\\"VALUE\\\"}]")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testJsonBooleanColumn(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement3="true")
        with self.assertRaisesRegexp(Exception, "No valid json string"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testIllegalQuoteColumn(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement3='''\\\"\\\\\\'t\\\'\\\"''')
        with self.assertRaisesRegexp(Exception, "Bad escape sequence in string."):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testAdapterNoteWithoutQuotesColumn(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement3="no quotes")
        with self.assertRaisesRegexp(Exception, "Syntax error: value, object or array expected."):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def testIncompleteAdapterNoteColumn(self):
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')
        self.createIllegalNotesAdapter(schemaName="ADAPTER", adapterName="ILLEGAL_NOTES_ADAPTER", replacement3="\\\"x")
        with self.assertRaisesRegexp(Exception, "Missing ',' or '}' in object declaration"):
            self.query('CREATE VIRTUAL SCHEMA VS_ILLEGAL_NOTES USING ADAPTER.ILLEGAL_NOTES_ADAPTER')
        self.query('DROP VIRTUAL SCHEMA IF EXISTS VS_ILLEGAL_NOTES CASCADE')

    def createIllegalNotesAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER", replacement="\\\"PLACEHOLDER\\\"", replacement2="\\\"PLACEHOLDER2\\\"", replacement3="\\\"PLACEHOLDER3\\\""):
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
                            "adapterNotes": "PLACEHOLDER",
                            "tables": [
                            {{
                                "adapterNotes": "PLACEHOLDER2",
                                "name": "DUMMY",
                                "columns": [{{
                                    "adapterNotes": "PLACEHOLDER3",
                                    "name": "KEY",
                                    "dataType": {{"type": "VARCHAR", "size": 100}}
                                }},{{
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 100}}
                                }}]
                            }}]
                        }}
                    }}
                    return json.dumps(res).replace("\\"PLACEHOLDER\\"","{replace}").replace("\\"PLACEHOLDER\\"","{replace}").replace("\\"PLACEHOLDER2\\"","{replace2}").replace("\\"PLACEHOLDER3\\"","{replace3}").encode('utf-8')
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
            ''').format(schema = schemaName, adapter = adapterName, replace = replacement, replace2 = replacement2, replace3 = replacement3))

class ViewPrivileges(VSchemaTest):

    def testGetConnection(self):
        self.createUser("foo", "foo")
        self.query('''CREATE SCHEMA IF NOT EXISTS SPOT4245''')
        self.query('''
            create or replace connection AC_FOOCONN to 'a' user 'b' identified by 'c'
        ''')
        self.createConnectionAdapter(schemaName="SPOT4245", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS SPOT4245VS CASCADE')

        self.query('CREATE VIRTUAL SCHEMA SPOT4245VS USING SPOT4245.FAST_ADAPTER')
        rows = self.query('''
            SELECT * from SPOT4245VS.DUMMY
            ''')
        self.assertRowsEqual([('password','a','b','c')],rows)
        self.query("OPEN SCHEMA SPOT4245")
        self.query("create or replace view SPOT4245.SPOT4245VIEW as SELECT * from SPOT4245VS.DUMMY")
        self.query("grant select on SPOT4245.SPOT4245VIEW to foo")
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('''select * from SPOT4245.SPOT4245VIEW''')
        self.assertRowsEqual([('password','a','b','c')],rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS SPOT4245VS CASCADE')
        self.query('''DROP SCHEMA IF EXISTS SPOT4245 CASCADE''')

    def testGetInput(self):
        self.createUser("foo", "foo")
        self.query('''CREATE SCHEMA IF NOT EXISTS SPOT4245''')
        self.query(udf.fixindent('''
            CREATE OR REPLACE python SCALAR SCRIPT
            spot42542script()
            RETURNS VARCHAR(200) AS
            def f():
                return "42"
            /
            '''))
        self.createImportAdapter(schemaName="SPOT4245", adapterName="FAST_ADAPTER")
        self.query('DROP VIRTUAL SCHEMA IF EXISTS SPOT4245VS CASCADE')

        self.query('CREATE VIRTUAL SCHEMA SPOT4245VS USING SPOT4245.FAST_ADAPTER')
        rows = self.query('''
            SELECT * from SPOT4245VS.DUMMY
            ''')
        self.assertRowsEqual([('42',)],rows)
        self.query("OPEN SCHEMA SPOT4245")
        self.query("create or replace view SPOT4245.SPOT4245VIEW as SELECT * from SPOT4245VS.DUMMY")
        self.query("grant select on SPOT4245.SPOT4245VIEW to foo")
        self.commit()
        foo_conn = self.getConnection('foo', 'foo')
        rows = foo_conn.query('''select * from SPOT4245.SPOT4245VIEW''')
        self.assertRowsEqual([('42', )],rows)
        self.query('DROP VIRTUAL SCHEMA IF EXISTS SPOT4245VS CASCADE')
        self.query('''DROP SCHEMA IF EXISTS SPOT4245 CASCADE''')

    def createConnectionAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER"):
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
                                    "name": "TYPE",
                                    "dataType": {{"type": "VARCHAR", "size": 200}}
                                }},{{
                                    "name": "HOST",
                                    "dataType": {{"type": "VARCHAR", "size": 200}}
                                }},{{
                                    "name": "CONN",
                                    "dataType": {{"type": "VARCHAR", "size": 200}}
                                }},{{
                                    "name": "PWD",
                                    "dataType": {{"type": "VARCHAR", "size": 200}}
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
                        "sql": "SELECT * FROM (VALUES ('PLACEHOLDER1', 'PLACEHOLDER2', 'PLACEHOLDER3', 'PLACEHOLDER4')) t"
                    }}
                    c = exa.get_connection('AC_FOOCONN')
                    return json.dumps(res).replace("PLACEHOLDER1",c.type).replace("PLACEHOLDER2",c.address).replace("PLACEHOLDER3",c.user).replace("PLACEHOLDER4",c.password).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

    def createImportAdapter(self, schemaName="ADAPTER", adapterName="FAST_ADAPTER"):
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
                                    "name": "VALUE",
                                    "dataType": {{"type": "VARCHAR", "size": 200}}
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
                        "sql": "SELECT * FROM (VALUES ('PLACEHOLDER1')) t"
                    }}
                    c = exa.import_script('spot42542script')
                    return json.dumps(res).replace("PLACEHOLDER1",c.f()).encode('utf-8')
                else:
                    raise ValueError('Unsupported callback')
            /
            ''').format(schema = schemaName, adapter = adapterName))

class ReportedBugs(VSchemaTest):
    setupDone = False

    def setUp(self):
        # TODO This is another ugly workaround for the problem that the framework doesn't offer us a query in classmethod setUpClass. Rewrite!
        if self.__class__.setupDone:
            self.query(''' CLOSE SCHEMA ''')
            return

        self.createJdbcAdapter()
        self.createNative()
        self.commit()  # We have to commit, otherwise the adapter won't see these tables
        self.createVirtualSchemaJdbc("VS1", "NATIVE", "ADAPTER.JDBC_ADAPTER", False)
        self.commit()
        self.query(''' CLOSE SCHEMA ''')
        self.__class__.setupDone = True

    def testEqualColumns(self):
        # Single Group
        self.compareWithNativeExtended('''
            select 1 from {v}.t WHERE t.A = t.A
        ''', ignoreOrder=True, explainResponse='''SELECT 1 FROM NATIVE.T''')
        self.assertExpectations()

    def testJoinWithSubselect(self):
        # Single Group
        self.compareWithNativeExtended('''
            SELECT 1 FROM {v}.t_nulls VS LEFT JOIN (SELECT DISTINCT DUMMY FROM SYS.DUAL) D ON VS.A=D.DUMMY
            LEFT JOIN SYS.DUAL D1 ON 1=1
        ''', ignoreOrder=True, explainResponse='''SELECT A FROM NATIVE.T_NULLS''')
        self.assertExpectations()

if __name__ == '__main__':
    udf.main()
