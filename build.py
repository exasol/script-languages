import subprocess

if __name__ == '__main__':
    command = "luigi --log-level INFO --local-scheduler --workers 5 " \
              "--module build_utils DockerBuild " \
              "--flavor-path build_utils/test/resources/test-flavor/ " \
              "--build-config-force-build"
    p=subprocess.Popen(args=command.split(" "))
    p.communicate()