import subprocess

if __name__ == '__main__':
    command = [
        "luigi-deps-tree",
        "--local-scheduler",
        "--module", "build_utils", "TestContainer",
        "--flavor-path", "build_utils/test/resources/test-flavor/",
        "--reuse-database",
        "--reuse-uploaded-release-container",
#        "--test-folders", '["python"]',
#        "--tests-to-execute", '["test_unicode_umlaute"]',
    ]
    p = subprocess.Popen(args=command)
    p.communicate()