import subprocess

if __name__ == '__main__':
    p=subprocess.Popen(args="pipenv run luigi --local-scheduler --module build_utils.workflow DockerBuild --FlavorConfig-flavor-path build_utils/test/resources/test-flavor/".split(" "))
    p.communicate()