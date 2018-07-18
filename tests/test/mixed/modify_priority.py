# -*- coding: utf-8 -*-
#!/usr/opt/bs-python-2.7/bin/python
#SPOT-2234

import os
import sys
import unittest
import time
import threading
import Queue

sys.path.append(os.path.realpath(__file__ + '/../../../lib'))

import udf
from udf import requires
import exatest

class VisiblePriorityModificationTest(udf.TestCase):

#    queue = Queue.Queue(0)

    def setUp(self):

        # create table
        self.query('DROP SCHEMA IF EXISTS TEST_SCHEMA CASCADE')
        self.query('CREATE SCHEMA TEST_SCHEMA')
        self.query('CREATE TABLE t1 (name VARCHAR(10), nr INTEGER PRIMARY KEY)')
        self.query("""INSERT INTO t1 VALUES ('u0', 0), ('u1', 1), ('u2', 2), ('u3', 3), ('u4', 4), ('u5', 5), ('u6', 6), ('u7', 7), ('u8', 8), ('u9', 9), ('u10', 10), ('u11', 11), ('u12', 12), ('u13', 13), ('u14', 14), ('u15', 15), ('u16', 16), ('u17', 17), ('u18', 18), ('u19', 19), ('u20', 20)""")
        self.commit()

        # create users with default priority MEDIUM
        self.createUser('U1', "u1")
        self.createUser('U2', "u2")


    def tearDown(self):
        # cleanup users
        self.query('DROP USER U1')
        self.query('DROP USER U2')

        # drop t1
        self.query('OPEN SCHEMA TEST_SCHEMA')
        self.query('DROP TABLE t1')
        self.query('DROP SCHEMA TEST_SCHEMA')

    def getConnection(self, username, password):
        client = exatest.ODBCClient('exatest')
        self.log.debug('connecting to DSN "exa" for user {username}'.format(username=username))
        client.connect(uid = username, pwd = password)
        return client

    def createUser(self, username, password):
        self.query('DROP USER IF EXISTS {username} CASCADE'.format(username=username) )
        self.query('CREATE USER {username} IDENTIFIED BY "{password}"'.format(username = username, password = password))

        # grant for user
        self.query('GRANT CREATE SESSION TO {username}'.format(username=username))
        self.query('GRANT SELECT ANY TABLE TO {username}'.format(username=username))
        self.query('GRANT SELECT, DELETE, UPDATE ON TABLE t1 TO {username}'.format(username=username))

        # default priority is MEDIUM
        self.setPriority(username, 'MEDIUM')
        self.commit()

    def setPriority(self,username, priority):
        self.query('GRANT PRIORITY {priority} TO {username}'.format(priority=priority, username=username))

    # return dictionary with username and priority, order by username ASC
    def mappedgetPriorityFromEXA_ALL_SESSIONS(self):
        result = self.query("""SELECT USER_NAME, PRIORITY FROM EXA_ALL_SESSIONS""")
        priorities = {}
        for row in result:
            priorities[row[0]] = row[1]
        return priorities

    # return dictionary with username and weight, order by username asc
    def mappedgetWeightFromEXA_RM_PROCESS_STATES(self):
        result = self.query("""SELECT USER_NAME, R.WEIGHT FROM "$EXA_SESSIONS_BASE" S, "$EXA_RM_PROCESS_STATES" R WHERE R.SESSION_ID = S.SESSION_ID""")
        weights = {}
        for row in result:
            weights[row[0]] = row[1]
        return weights

    def userSessionTwoUsers(self, conn, username, queue):
        # execute select on t1 for several times
        conn.query('OPEN SCHEMA TEST_SCHEMA')
        conn.query('ALTER SESSION SET QUERY_TIMEOUT=5')

        queue.get()
        conn.query('SELECT COUNT(*) from T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1')


    def userSessionOneUser(self, conn, username, queue):
        conn.query('OPEN SCHEMA TEST_SCHEMA')
        conn.query('ALTER SESSION SET QUERY_TIMEOUT=5')

        item = queue.get()
        conn.query('SELECT COUNT(*) from T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1')

        item = queue.get()
        conn.query('SELECT COUNT(*) from T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1')

        item = queue.get()
        conn.query('SELECT COUNT(*) from T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1, T1')


    def testPriorityTwoUsers(self):
        queue1 = Queue.Queue(0)
        queue2 = Queue.Queue(0)

        # create connections and threads for U1 and U2
        connectionu1= self.getConnection('U1', "u1")
        connectionu2= self.getConnection('U2', "u2")
        userThread1 = threading.Thread(target=self.userSessionTwoUsers, args = (connectionu1,"U1", queue1))
        userThread2 = threading.Thread(target=self.userSessionTwoUsers, args = (connectionu2,"U2", queue2))

        # start
        userThread1.start()
        userThread2.start()

        queue1.put("item")
        queue2.put("item")

        # modify priority 1
        self.setPriority('U2', "LOW")
        self.setPriority('U1', "HIGH")
        self.commit()

        loopCount = 0
        priorsOk = False
        weightsOk = False

        while True:
            if(priorsOk == False):
                priors = self.mappedgetPriorityFromEXA_ALL_SESSIONS()
            if(weightsOk == False):
                weights = self.mappedgetWeightFromEXA_RM_PROCESS_STATES()

            if(loopCount > 30):
                print("timeout")
                break
            loopCount = loopCount + 1

            if(priorsOk == True and weightsOk == True):
                break
            else:
                if(weights['U1'] != None and weights['U2'] != None):
                    weightsOk = True
                if(priors['U1'] == 'HIGH' and priors['U2'] == 'LOW'):
                    priorsOk = True

                if(priorsOk != True):
                    print("priors not yet ok: " + str(priors))
                if(weightsOk != True):
                    print("weights not yet ok" + str(weights))

                if(priorsOk != True or weightsOk != True):
                    time.sleep(0.1)
                    continue

        self.assertTrue(priors.has_key('U1'))
        self.assertEqual(priors['U1'], 'HIGH')
        self.assertTrue(priors.has_key('U2'))
        self.assertEqual(priors['U2'], 'LOW')

        weightsSum = weights['U1'] + weights['U2'] + weights['SYS']
        self.assertTrue(weightsSum <= 101 or weightsSum >= 99)

        # join
        userThread1.join()
        userThread2.join()

        connectionu1.rollback()
        connectionu1.close()
        connectionu2.rollback()
        connectionu2.close()


    def testPriorityOneUser(self):
        queue = Queue.Queue(0)
        connectionu1= self.getConnection('U1', "u1")
        userThread = threading.Thread(target=self.userSessionOneUser, args = (connectionu1,"U1", queue))
        userThread.start()

        # get priors and weights in Query 1
        queue.put("item")
        time.sleep(1)

        loopCount = 0
        priorsOk = False
        weightsOk = False

        while True:

            if(priorsOk == False):
                priors1 = self.mappedgetPriorityFromEXA_ALL_SESSIONS()
            if(weightsOk == False):
                weights1 = self.mappedgetWeightFromEXA_RM_PROCESS_STATES()
            if(loopCount >= 10):
                print("timeout...")
                break
            loopCount = loopCount + 1

            if(priorsOk == True and weightsOk == True):
                break
            else:
                if(weights1.has_key('U1') or weights1['U1'] != None):
                    weightsOk = True
                if(priors1.has_key('U1')):
                    priorsOk = True

                if(priorsOk != True):
                    print("priors1 not yet ok: " + str(priors1))

                if(weightsOk != True):
                    print("weights1 not yet ok: " + str(weights1))
                if(priorsOk != True or weightsOk != True):
                    time.sleep(0.1)
                    continue

        # assertions
        self.assertTrue(priors1.has_key('U1'))
        self.assertEqual(priors1['U1'], 'MEDIUM')
        self.assertTrue(weights1.has_key('U1'))
        weightsSum = weights1['U1'] + weights1['SYS']
        self.assertTrue( weightsSum <= 101 or weightsSum >= 99)


        # get priors and weights in Query 2 first time
        queue.put("item")
        loopCount = 0
        priorsOk = False
        weightsOk = False

        while True:

            if(priorsOk == False):
                priors2 = self.mappedgetPriorityFromEXA_ALL_SESSIONS()
            if(weightsOk == False):
                weights2 = self.mappedgetWeightFromEXA_RM_PROCESS_STATES()
            if(loopCount >= 10):
                print("timeout...")
                break
            loopCount = loopCount + 1

            if(priorsOk == True and weightsOk == True):
                break
            else:
                if(weights2.has_key('U1')):
                    weightsOk = True
                if(priors2['U1'] == 'MEDIUM'):
                    priorsOk = True

                if(priorsOk != True):
                    print("priors2 not yet ok: " + str(priors2))
                if(weightsOk != True):
                    print("weights2 not yet ok: " + str(weights2))
                if(priorsOk != True or weightsOk != True):
                    time.sleep(0.1)
                    continue

        # assertions
        self.assertTrue(priors2.has_key('U1'))
        self.assertEqual(priors2['U1'], 'MEDIUM')
        self.assertTrue(weights2.has_key('U1'))
        weightsSum = weights2['U1'] + weights2['SYS']
        self.assertTrue( weightsSum <= 101 or weightsSum >= 99)
        self.assertEqual(priors1, priors2)
        self.assertEqual(weights1, weights2)

        # modify priority U1
        self.setPriority('U1', "LOW")
        self.commit()

        # get priors and weights in Query 2 second time
        loopCount = 0
        priorsOk = False
        weightsOk = False

        while True:

            if(priorsOk == False):
                priors3 = self.mappedgetPriorityFromEXA_ALL_SESSIONS()
            if(weightsOk == False):
                weights3 = self.mappedgetWeightFromEXA_RM_PROCESS_STATES()
            if(loopCount >= 10):
                print("timeout...")
                break
            loopCount = loopCount + 1

            if(priorsOk == True and weightsOk == True):
                break
            else:
                if(weights3.has_key('U1')):
                    weightsOk = True
                if(priors3['U1'] == 'LOW'):
                    priorsOk = True

                if(priorsOk != True):
                    print("priors3 not yet ok: " + str(priors3))

                if(weightsOk != True):
                    print("weights3 not yet ok: " + str(weights3))
                if(priorsOk != True or weightsOk != True):
                    time.sleep(0.1)
                    continue

        # assertions
        self.assertTrue(priors3.has_key('U1'))
        self.assertEqual(priors3['U1'], 'LOW')
        self.assertTrue(weights3.has_key('U1'))
        weightsSum = weights3['U1'] + weights3['SYS']
        self.assertTrue( weightsSum <= 101 or weightsSum >= 99)
        self.assertNotEqual(priors2, priors3)
        self.assertNotEqual(weights2, weights3)

        # get priors and weights in Query 3
        queue.put("item")
        loopCount = 0
        priorsOk = False
        weightsOk = False

        while True:

            if(priorsOk == False):
                priors4 = self.mappedgetPriorityFromEXA_ALL_SESSIONS()
            if(weightsOk == False):
                weights4 = self.mappedgetWeightFromEXA_RM_PROCESS_STATES()
            if(loopCount >= 10):
                print("timeout...")
                break
            loopCount = loopCount + 1

            if(priorsOk == True and weightsOk == True):
                break
            else:
                if(weights4.has_key('U1')):
                    weightsOk = True
                if(priors4['U1'] == 'LOW'):
                    priorsOk = True

                if(priorsOk != True):
                    print("priors4 not yet ok: " + str(priors4))

                if(weightsOk != True):
                    print("weights4 not yet ok: " + str(weights4))
                if(priorsOk != True or weightsOk != True):
                    time.sleep(0.1)
                    continue

        # assertions
        self.assertTrue(priors4.has_key('U1'))
        self.assertEqual(priors4['U1'], 'LOW')
        self.assertTrue(weights4.has_key('U1'))
        weightsSum = weights4['U1'] + weights4['SYS']
        self.assertTrue( weightsSum <= 101 or weightsSum >= 99)
        self.assertEqual(priors3, priors4)
        self.assertEqual(weights3, weights4)

        userThread.join()

        connectionu1.rollback()
        connectionu1.close()

if __name__ == '__main__':
    udf.main()
