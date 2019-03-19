import base64
import os
import shutil
import tempfile
import unittest
from build_utils.lib.utils.directory_hasher import FileDirectoryListHasher


class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        level1 = 5
        level2 = 5
        level3 = 5
        level4 = 5
        level5 = 5
        for i1 in range(level1):
            for i2 in range(level2):
                for i3 in range(level3):
                    for i4 in range(level4):
                        path = "%s/level1_%s/level2_%s/level3_%s/level4_%s/" % (self.test_dir,i1, i2, i3, i4)
                        os.makedirs(path)
                        for i5 in range(level5):
                            file = "%s/level5_file_%s" % (path, i5)
                            with open(file, mode="xt") as f:
                                f.write(file)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_something(self):
        # print(FileDirectoryListHasher(hashfunc="sha256",hash_file_names=False, hash_directory_names=False).hash([self.test_dir]))
        # print(FileDirectoryListHasher(hashfunc="sha256",hash_file_names=False, hash_directory_names=True).hash([self.test_dir]))
        # print(FileDirectoryListHasher(hashfunc="sha256",hash_file_names=True, hash_directory_names=False).hash([self.test_dir]))
        # print(FileDirectoryListHasher(hashfunc="sha256",hash_file_names=True, hash_directory_names=True).hash([self.test_dir]))
        # print(FileDirectoryListHasher(hashfunc="sha256",hash_file_names=True, hash_directory_names=True, hash_permissions=True).hash([self.test_dir]))
        # print()
        # print("==============================================")
        # print()
        # print(FileDirectoryListHasher(hashfunc="sha256").hash([self.test_dir]))
        # print(FileDirectoryListHasher(hashfunc="sha256").hash([self.test_dir]).hex())
        print(base64.urlsafe_b64encode(FileDirectoryListHasher(hashfunc="sha256").hash(["."])))


if __name__ == '__main__':
    unittest.main()
