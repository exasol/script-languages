#!/usr/bin/env python2.7
# encoding: utf8

import os
import sys
import time
import math
sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf

class AbstractPerformanceTest(udf.TestCase):

    def create_schema(self):
        self.schema = self.__class__.__name__
        self.query('DROP SCHEMA %s CASCADE'%self.schema, ignore_errors=True)
        self.query('CREATE SCHEMA %s'%self.schema)
        self.query('OPEN SCHEMA %s'%self.schema)

    def generate_data_linear(self,multiplier,base=1000):
        self.query('CREATE TABLE BATCH (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        for i in range(base):
            self.query('''INSERT INTO BATCH values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                        123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')
        max_unions = 10
        for i in range(int(math.floor(multiplier/max_unions))):
            selects_for_multiplier = ['''select * from BATCH''' for i in range(max_unions)]
            union_query_for_multiplier = " union all ".join(selects_for_multiplier)
            self.query('''INSERT INTO T %s;'''%union_query_for_multiplier)
        if multiplier % max_unions > 0:
            selects_for_multiplier = ['''select * from BATCH''' for i in range(multiplier % max_unions)]
            union_query_for_multiplier = " union all ".join(selects_for_multiplier)
            self.query('''INSERT INTO T %s;'''%union_query_for_multiplier)

    def generate_data_exponential(self,rounds,exponent=10,base=1000):
        self.query('CREATE TABLE T (intVal DECIMAL(9,0), longVal DECIMAL(18,0), bigdecimalVal DECIMAL(36,0), decimalVal DECIMAL(9,2), \
                    doubleVal DOUBLE, doubleIntVal DOUBLE, stringVal VARCHAR(100), booleanVal BOOLEAN, dateVal DATE, timestampVal TIMESTAMP)')
        for i in range(base):
            self.query('''INSERT INTO T values (123456789, 123456789123456789, 123456789123456789123456789123456789, 1234567.12, \
                        123456789.123, 15.0, 'string#String!12345', true, '2014-05-21', '2014-05-21 15:13:30.123')''')
        selects_for_exponent = ['''select * from T''' for i in range(exponent)]
        union_query_for_exponent = " union all ".join(selects_for_exponent)
        for i in range(rounds):
            self.query('''INSERT INTO T %s;'''%union_query_for_exponent)

    def cleanup(self,schema):
        self.query('DROP SCHEMA %s CASCADE'%schema, ignore_errors=True)

    def run_test(self, runs, warmup, max_deviation, query):
        connection = self.getConnection(self.user,self.password)
        under_test_mean_elapsed_time,under_test_variance_elapsed_time,\
        under_test_max_elapsed_time,under_test_min_elapsed_time=\
                    self.run_queries(connection,"under_test", runs, warmup, query)
        connection.close()

        connection = self.getConnection(self.user,self.password)
        connection.query("ALTER SESSION SET script_languages='PYTHON=builtin_python PYTHON3=builtin_python3 JAVA=builtin_java R=builtin_r'")
        builtin_mean_elapsed_time,builtin_variance_elapsed_time,\
        builtin_max_elapsed_time,builtin_min_elapsed_time=\
                self.run_queries(connection,"builtin_python", runs, warmup, query)
        connection.close()

        deviation = 100-builtin_mean_elapsed_time/under_test_mean_elapsed_time*100
        print("deviation:",deviation)
        print("under_test_mean_elapsed_time:",under_test_mean_elapsed_time)
        print("under_test_variance_elapsed_time:",under_test_variance_elapsed_time)
        print("under_test_max_elapsed_time:",under_test_max_elapsed_time)
        print("under_test_min_elapsed_time:",under_test_min_elapsed_time)
        print("builtin_mean_elapsed_time:",builtin_mean_elapsed_time)
        print("builtin_variance_elapsed_time:",builtin_variance_elapsed_time)
        print("builtin_max_elapsed_time:",builtin_max_elapsed_time)
        print("builtin_min_elapsed_time:",builtin_min_elapsed_time)
        sys.stdout.flush()
        self.assertLessEqual(deviation,max_deviation,"Deviation of mean elapsed time %s greater than %s (under_test: %s, builtin: %s)"% \
                (deviation,max_deviation,under_test_mean_elapsed_time,builtin_mean_elapsed_time))
        self.assertGreaterEqual(deviation,-max_deviation,"Deviation of mean elapsed time %s less than %s (under_test: %s, builtin: %s)"% \
                (deviation,-max_deviation,under_test_mean_elapsed_time,builtin_mean_elapsed_time))

    def run_queries(self,connection, test_name, runs, warmup, query):
        connection.query("alter session set query_cache='off'")
        connection.query('OPEN SCHEMA %s'%self.schema)
        print("WARMUP")
        for i in range(warmup):
            elapsed=self.measure_query(connection, query)
            print(test_name,i,elapsed)
        print("MEASUREMENT")
        elapsed_times = []
        for i in range(runs):
            elapsed=self.measure_query(connection, query)
            elapsed_times.append(elapsed)
            print(test_name,i,elapsed)
        mean = sum(elapsed_times)/(len(elapsed_times))
        max_elapsed_times=max(elapsed_times)
        min_elapsed_times=min(elapsed_times)
        variance_from_mean = sum([(xi - mean)**2 for xi in elapsed_times]) / (len(elapsed_times))
        return mean, variance_from_mean, max_elapsed_times,min_elapsed_times

    def measure_query(self, conn, query):
        start = time.time()
        rows = conn.query(query)
        end = time.time()
        return end-start


# vim: ts=4:sts=4:sw=4:et:fdm=indent

