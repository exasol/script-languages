import subprocess

if __name__ == '__main__':
    command = "luigi --log-level INFO --local-scheduler --workers 5 " \
              "--module build_utils TestContainer " \
              "--flavor-path build_utils/test/resources/test-flavor/ " \
              "--reuse-database"
    p=subprocess.Popen(args=command.split(" "))
    p.communicate()