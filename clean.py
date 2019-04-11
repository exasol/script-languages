import subprocess

if __name__ == '__main__':
    command = "luigi --log-level INFO --local-scheduler --workers 5 " \
              "--module build_utils CleanImages " \
              "--docker-config-repository-name exasol/script-language-container"
    p=subprocess.Popen(args=command.split(" "))
    p.communicate()