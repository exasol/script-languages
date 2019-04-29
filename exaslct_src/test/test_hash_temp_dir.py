import base64
import os
import shutil
import tempfile
import unittest

from exaslct_src.lib.utils.file_directory_list_hasher import FileDirectoryListHasher

TEST_FILE = "/tmp/SEFQWEFWQEHDUWEFDGZWGDZWEFDUWESGRFUDWEGFUDWAFGWAZESGFDWZA"


class HashTempDirTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir1 = self.temp_dir + "/test1"
        self.generate_test_dir(self.test_dir1)
        self.test_dir2 = self.temp_dir + "/test2"
        self.generate_test_dir(self.test_dir2)
        with open(TEST_FILE, "wt") as f:
            f.write("test")

    def generate_test_dir(self, test_dir):
        level1 = 5
        level2 = 5
        level3 = 5
        level4 = 5
        level5 = 5
        for i1 in range(level1):
            for i2 in range(level2):
                for i3 in range(level3):
                    for i4 in range(level4):
                        path = "/level0/level1_%s/level2_%s/level3_%s/level4_%s/" \
                               % (i1, i2, i3, i4)
                        os.makedirs(test_dir + path)
                        os.makedirs(test_dir + path + "test")
                        for i5 in range(level5):
                            file = "%s/level5_file_%s" % (path, i5)
                            with open(test_dir + file, mode="wt") as f:
                                f.write(file)

    def tearDown(self):
        os.remove(TEST_FILE)
        shutil.rmtree(self.temp_dir)

    def test_file_content_only_fixed_hash(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=False,
                                         hash_directory_names=False,
                                         hash_file_names=False)
        hash = hasher.hash([TEST_FILE])
        ascii_hash = base64.b32encode(hash).decode("ASCII")
        self.assertEqual("SVGVUSP5ODM3RPG3GXJFEJTYFGKX67XX7JWHJ6EEDG64L2BCBH2A====", ascii_hash)

    def test_file_with_path(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=False,
                                         hash_directory_names=True,
                                         hash_file_names=True)
        hash = hasher.hash([TEST_FILE])
        ascii_hash = base64.b32encode(hash).decode("ASCII")
        self.assertEqual("AOBF7HELTYAPHBEW6HQQ74N3BKGSNZTJXB4MTOEROHO5VH6YYJOA====", ascii_hash)

    def test_directory_with_relative_paths_fixed_hash(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=True,
                                         hash_directory_names=True,
                                         hash_file_names=True)
        hash = hasher.hash([self.test_dir1])
        ascii_hash = base64.b32encode(hash).decode("ASCII")
        self.assertEqual("VIN3VCPDX7DAC4GD37IDF4KQTCDNNH72QV5PARVGGQ4OMB4DZTLA====", ascii_hash)

    def test_directory_content_only_fixed_hash(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=False,
                                         hash_directory_names=False,
                                         hash_file_names=False)
        hash = hasher.hash([self.test_dir1])
        ascii_hash = base64.b32encode(hash).decode("ASCII")
        self.assertEqual("TM2V22T326TCTLQ537BZAOR3I5NVHXE6IDJ4TXPCJPTUGDTI5WYQ====", ascii_hash)

    def test_directory_with_relative_paths_equal(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=True,
                                         hash_directory_names=True,
                                         hash_file_names=True)
        hash1 = hasher.hash([self.test_dir1])
        hash2 = hasher.hash([self.test_dir2])
        ascii_hash1 = base64.b32encode(hash1).decode("ASCII")
        ascii_hash2 = base64.b32encode(hash2).decode("ASCII")
        self.assertEqual(ascii_hash1, ascii_hash2)

    def test_directory_without_relative_paths_not_equal(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=False,
                                         hash_directory_names=True,
                                         hash_file_names=True)
        hash1 = hasher.hash([self.test_dir1])
        hash2 = hasher.hash([self.test_dir2])
        ascii_hash1 = base64.b32encode(hash1).decode("ASCII")
        ascii_hash2 = base64.b32encode(hash2).decode("ASCII")
        self.assertNotEqual(ascii_hash1, ascii_hash2)

    def test_directory_content_only_equal(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=False,
                                         hash_directory_names=False,
                                         hash_file_names=False)
        hash1 = hasher.hash([self.test_dir1])
        hash2 = hasher.hash([self.test_dir2])
        ascii_hash1 = base64.b32encode(hash1).decode("ASCII")
        ascii_hash2 = base64.b32encode(hash2).decode("ASCII")
        self.assertEqual(ascii_hash1, ascii_hash2)

    def test_directory_relative_paths_equal(self):
        hasher = FileDirectoryListHasher(hashfunc="sha256",
                                         use_relative_paths=True,
                                         hash_directory_names=False,
                                         hash_file_names=False)
        hash1 = hasher.hash([self.test_dir1])
        hash2 = hasher.hash([self.test_dir2])
        ascii_hash1 = base64.b32encode(hash1).decode("ASCII")
        ascii_hash2 = base64.b32encode(hash2).decode("ASCII")
        self.assertEqual(ascii_hash1, ascii_hash2)

    def test_directory_content_only_not_equal_to_with_paths(self):
        hasher_content_only = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=False,
                                    hash_directory_names=False,
                                    hash_file_names=False)
        hasher_with_paths = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=True,
                                    hash_directory_names=True,
                                    hash_file_names=True)
        hash1_content_only = hasher_content_only.hash([self.test_dir1])
        hash2_with_paths = hasher_with_paths.hash([self.test_dir2])
        ascii_hash1_content_only = base64.b32encode(hash1_content_only).decode("ASCII")
        ascii_hash2_with_paths = base64.b32encode(hash2_with_paths).decode("ASCII")
        self.assertNotEqual(ascii_hash1_content_only, ascii_hash2_with_paths)

    def test_directory_content_only_not_equal_to_dir_names(self):
        hasher_content_only = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=False,
                                    hash_directory_names=False,
                                    hash_file_names=False)
        hasher_with_paths = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=False,
                                    hash_directory_names=True,
                                    hash_file_names=False)
        hash1_content_only = hasher_content_only.hash([self.test_dir1])
        hash2_with_paths = hasher_with_paths.hash([self.test_dir1])
        ascii_hash1_content_only = base64.b32encode(hash1_content_only).decode("ASCII")
        ascii_hash2_with_paths = base64.b32encode(hash2_with_paths).decode("ASCII")
        self.assertNotEqual(ascii_hash1_content_only, ascii_hash2_with_paths)

    def test_directory_content_only_not_equal_to_file_names(self):
        hasher_content_only = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=False,
                                    hash_directory_names=False,
                                    hash_file_names=False)
        hasher_with_paths = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=False,
                                    hash_directory_names=False,
                                    hash_file_names=True)
        hash1_content_only = hasher_content_only.hash([self.test_dir1])
        hash2_with_paths = hasher_with_paths.hash([self.test_dir1])
        ascii_hash1_content_only = base64.b32encode(hash1_content_only).decode("ASCII")
        ascii_hash2_with_paths = base64.b32encode(hash2_with_paths).decode("ASCII")
        self.assertNotEqual(ascii_hash1_content_only, ascii_hash2_with_paths)

    def test_directory_file_names_not_equal_to_dir_names(self):
        hasher_content_only = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=True,
                                    hash_directory_names=False,
                                    hash_file_names=True)
        hasher_with_paths = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    use_relative_paths=True,
                                    hash_directory_names=True,
                                    hash_file_names=False)
        hash1_content_only = hasher_content_only.hash([self.test_dir1])
        hash2_with_paths = hasher_with_paths.hash([self.test_dir2])
        ascii_hash1_content_only = base64.b32encode(hash1_content_only).decode("ASCII")
        ascii_hash2_with_paths = base64.b32encode(hash2_with_paths).decode("ASCII")
        self.assertNotEqual(ascii_hash1_content_only, ascii_hash2_with_paths)


if __name__ == '__main__':
    unittest.main()
