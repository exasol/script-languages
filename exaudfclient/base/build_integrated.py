import sys


def build_integrated(target, source):
    assert(len(target) == 1 and len(source) > 0)
    builddir = '.'
    RE='/usr'

    output = []
    for fname in source:
        fname_short = str(fname)
        if fname_short.startswith(builddir):
            fname_short = fname_short[len(builddir):]
        fvar = 'integrated_' + fname_short.lower().replace('.', '_').replace('/', '_')
        flines = []
        with open(str(fname)) as fname_file:
            for line in fname_file:
                line = line.replace(", PACKAGE='exascript_r'", '')
                line = line.replace('\\', '\\\\')
                line = line.replace(r'"', r'\"')
                line = line.replace('\r', r'\r')
                line = line.replace('\n', r'\n')
                line = line.replace('RUNTIME_PATH', RE)
                flines.append(line)
        output.append('static const char *' + fvar + ' = "' + ''.join(flines) + '";\n')
    with open(str(target[0]), 'w') as fd:
        fd.write(''.join(output))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: build_integrated.py target source_1 source_2 ...')
        sys.exit(1)
    build_integrated(sys.argv[1:2], sys.argv[2:])
