import subprocess

if __name__ == '__main__':
    command = [
        "luigi",
        "--execution-summary-summary-length","0",
        "--log-level", "INFO",
        "--local-scheduler",
        "--workers", "2",
        "--module", "build_utils", "TestContainer",
        "--flavor-path", "build_utils/test/resources/test-flavor/",
        "--log-config-log-task-is-still-running",
        "--SpawnTestDockerDatabase-db-startup-timeout-in-seconds",str(60*10),
        # "--reuse-database",
        # "--reuse-uploaded-release-container",
        "--ignore-flavor-test-config",
        "--test-files", '["python/unicode.py"]'
    ]
    p = subprocess.Popen(args=command)
    p.communicate()
