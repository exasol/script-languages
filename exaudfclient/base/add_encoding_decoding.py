import os
import sys
from pathlib import Path

CONVERT_TO_BYTES = """
decodeUTF8 = lambda x: x.decode(encoding='utf-8') if isinstance(x, bytes) else x
encodeUTF8 = lambda x: x.encode(encoding='utf-8') if isinstance(x, str) else x
"""

IDENTITY = """
decodeUTF8 = lambda x: x
encodeUTF8 = lambda x: x
"""

def add_encoding_decoding(target):
    if "SWIG_STRING_AS_BYTES_ENABLED" in os.environ:
        decoding_encdecoding = CONVERT_TO_BYTES
    else:
        decoding_encdecoding = IDENTITY

    with open(target, "wt", encoding="utf-8") as f:
        f.write(decoding_encdecoding)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: add_encoding_decoding.py target')
        sys.exit(1)
    add_encoding_decoding(sys.argv[1])
