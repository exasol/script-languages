
from contextlib import contextmanager
import os
import shutil
import tempfile

@contextmanager
def tempdir():
    tmp = tempfile.mkdtemp()
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp)

@contextmanager
def chdir(newdir):
    olddir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(olddir)

# vim: ts=4:sts=4:sw=4:et:fdm=indent

