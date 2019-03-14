import subprocess

if __name__ == '__main__':
    p=subprocess.Popen(args="luigi --log-level INFO --local-scheduler --workers 3 --module build_utils ReleaseContainer --flavor-path build_utils/test/resources/test-flavor/".split(" "))
    p.communicate()