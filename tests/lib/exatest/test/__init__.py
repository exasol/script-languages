'''Test unittest extensions with unittest'''

import cStringIO as StringIO
import contextlib
import sys
import unittest


def _print_output(output):
    print >> sys.stderr, '\n', '>' * 70
    print >> sys.stderr, output
    print >> sys.stderr, '<' * 70

@contextlib.contextmanager
def selftest(module, debug=False):
    '''Context manager to run unittests of unittest extensions in module.

    If debug is False, print test output only if exceptions are raised.

    Usage:
        
        class SefTest(unittest.TestCase):

            def test_metatest(self):
                class Module:
                    class Test(unittest.TestCase):
                        def test_fail(self):
                            self.fail()

                with selftest(Module) as result:
                    self.assertFalse(result.wasSucessful())
    '''
    try:
        stream = StringIO.StringIO()
        result = unittest.main(module=module,
                testRunner=unittest.TextTestRunner(stream=stream, verbosity=2),
                argv=sys.argv[:1], exit=False).result
        result.output = stream.getvalue()
        yield result
    except:
        _print_output(stream.getvalue())
        raise
    else:
        if debug:
            _print_output(stream.getvalue())


# vim: ts=4:sts=4:sw=4:et:fdm=indent
