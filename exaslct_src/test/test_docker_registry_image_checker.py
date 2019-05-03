import unittest

from docker.errors import DockerException

from exaslct_src.lib.docker.docker_registry_image_checker import DockerRegistryImageChecker


class MyTestCase(unittest.TestCase):

    def test_pull_success(self):
        image = "index.docker.io/registry:latest"
        checker = DockerRegistryImageChecker()
        exists = checker.check(image=image)
        print(exists)

    def test_pull_fail_with_DockerException(self):
        image = "index.docker.io/registry:abc"
        checker = DockerRegistryImageChecker()
        exists = lambda: checker.check(image=image)
        self.assertRaises(DockerException, exists)


if __name__ == '__main__':
    unittest.main()
