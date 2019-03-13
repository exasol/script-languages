import subprocess

if __name__ == '__main__':
    p=subprocess.Popen(args="luigi --log-level INFO --local-scheduler --module build_utils.workflow DockerBuild --build-config-force-build --flavor-path build_utils/test/resources/test-flavor/".split(" "))
    p.communicate()