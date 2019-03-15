import subprocess

if __name__ == '__main__':
    command = "luigi --log-level INFO --local-scheduler --workers 3 " \
              "--module build_utils DockerPush " \
              "--docker-config-repository exasol/script-language-container " \
              "--docker-config-username user " \
              "--docker-config-password password " \
              "--flavor-path build_utils/test/resources/test-flavor/"
    p=subprocess.Popen(args=command.split(" "))
    p.communicate()