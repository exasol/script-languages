# absolute_import is a mandatory feature in 2.7, but without this line, it does not work:

from __future__ import absolute_import 

import sys
import threading


__all__ = ['Thread', 'ThreadAliveError']

class ThreadAliveError(Exception): pass

class Thread(threading.Thread):
    '''Replacement of threading.Thread for use in TestCase.

    When join'ed, exceptions and assertion are collected.

    Usage:

        class Test(TestCase):
            def thread_fct(self, ...):
                while not self.shutdown_requested():
                    # do something
                self.assertTrue(...)

            def test_with_threads(self):
                t = Thread(target=self.thread_fct, name='thread-name')
                t.start()
                # do something
                t.shutdown() # optional
                t.join(5) # required to collect assertions
    '''

    def __init__(self, *args, **kwargs):
        super(Thread, self).__init__(*args, **kwargs)
        self.daemon = True
        self.__exc_info = None
        self.__shutdown_event = threading.Event()

    def run(self):
        try:
            super(Thread, self).run()
        except Exception:
            self.__exc_info = sys.exc_info()

    def join(self, timeout=None):
        super(Thread, self).join(timeout)
        #if self.is_alive():
        #    raise ThreadAliveError("thread '%s' is alive" % self.name)
        if self.__exc_info is not None:
            cls, msg, tb = self.__exc_info
            self.__exc_info = None
            raise cls("in thread '%s': %s" % (self.name, msg)), None, tb

    def shutdown(self):
        self.__shutdown_event.set()

    def shutdown_requested(self):
        return self.__shutdown_event.is_set()

# vim: ts=4:sts=4:sw=4:et:fdm=indent
