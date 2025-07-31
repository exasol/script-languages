"""
This script replaces the function `swig_import_helper`
in the automatically generated `exascript_python.py` with a simple import statement.
This is needed for Python 3.12, as the automatically generated code uses the library `imp`
which was removed in this Python version.
#TODO: Update Swig to a new version which produces code compatible with Python 3.12
"""
import sys
from pathlib import Path

old_function = """
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_exascript_python', [dirname(__file__)])
        except ImportError:
            import _exascript_python
            return _exascript_python
        if fp is not None:
            try:
                _mod = imp.load_module('_exascript_python', fp, pathname, description)
            finally:
                fp.close()
            return _mod
    _exascript_python = swig_import_helper()
    del swig_import_helper
"""

new_function = """
    import _exascript_python
"""

def replace_swig_import_helper(target, source):
    source_code = Path(source).read_text()

    assert old_function in source_code

    new_source_code = source_code.replace(old_function, new_function)

    with open(target, 'w') as f:
        f.write(new_source_code)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: replace_swig_import_helper.py target source')
        sys.exit(1)
    print(f"replacing function swig_import_helper() using source={sys.argv[2]} and target={sys.argv[1]}")
    replace_swig_import_helper(sys.argv[1], sys.argv[2])
