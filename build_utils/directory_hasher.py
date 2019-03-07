import os
import hashlib
import re
import stat
from typing import List

import humanfriendly
from multiprocessing import Pool

HASH_FUNCTIONS = {
    'md5': hashlib.md5,
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
    'sha512': hashlib.sha512
}


class FileDirectoryListHasher:

    def __init__(self,
                 hashfunc: str = 'md5',
                 followlinks: bool = False,
                 hash_file_names: bool = False,
                 hash_permissions: bool = False,
                 hash_directory_names: bool = False,
                 excluded_directories=None,
                 excluded_files=None,
                 excluded_extensions=None,
                 blocksize: str = "64kb",
                 workers: int = 4
                 ):

        self.workers = workers
        self.excluded_files = excluded_files
        self.excluded_extensions = excluded_extensions
        self.excluded_directories = excluded_directories
        self.hash_directory_names = hash_directory_names
        self.hash_permissions = hash_permissions
        self.hash_file_names = hash_file_names
        self.followlinks = followlinks

        self.hash_func = HASH_FUNCTIONS.get(hashfunc)
        if not self.hash_func:
            raise NotImplementedError('{} not implemented.'.format(hashfunc))

        if not self.excluded_files:
            self.excluded_files = []

        if not self.excluded_directories:
            self.excluded_directories = []

        if not self.excluded_extensions:
            self.excluded_extensions = []

        self.path_hasher = PathHasher(hashfunc, hash_permissions)
        self.file_content_hasher = FileContentHasher(hashfunc, blocksize)

    def hash(self, files_and_directories: List[str]) -> bytes:
        collected_directories = []
        collected_files = []
        if not isinstance(files_and_directories, List):
            raise Exception("List with paths expected and not '%s' with type %s"
                            % (files_and_directories, type(files_and_directories)))
        for file_or_directory in files_and_directories:
            if os.path.isdir(file_or_directory):
                self.collect_files_and_directories(file_or_directory, collected_directories, collected_files)
            elif os.path.isfile(file_or_directory):
                collected_files.append(file_or_directory)
            else:
                raise FileNotFoundError("Could not find file or directory %s" % file_or_directory)
        hashes = self.compute_hashes(collected_directories, collected_files)
        return self._reduce_hash(hashes)

    def compute_hashes(self, collected_directories, collected_files):
        collected_directories = sorted(collected_directories)
        collected_files = sorted(collected_files)
        pool = Pool(processes=self.workers)
        if self.hash_directory_names:
            file_path_hashes_of_directories = \
                pool.map_async(self.path_hasher.hash, collected_directories, chunksize=2)
        else:
            file_path_hashes_of_directories = None
        if self.hash_file_names:
            file_path_hashes_of_files = \
                pool.map_async(self.path_hasher.hash, collected_files, chunksize=2)
        else:
            file_path_hashes_of_files = None
        file_content_hashes = \
            pool.map_async(self.file_content_hasher.hash, collected_files, chunksize=2)
        hashes = []
        hashes.extend(file_content_hashes.get())
        if self.hash_directory_names:
            hashes.extend(file_path_hashes_of_directories.get())
        if self.hash_file_names:
            hashes.extend(file_path_hashes_of_files.get())
        return hashes

    def has_excluded_extension(self, f: str):
        return f.split('.')[-1:][0] in self.excluded_extensions

    def is_excluded_file(self, f: str):
        return f in self.excluded_files

    def is_excluded_directory(self, f: str):
        return f in self.excluded_directories

    def collect_files_and_directories(
            self, directory_name,
            collected_directories: List[str],
            collected_files: List[str]):
        if self.hash_directory_names:
            collected_directories.append(directory_name)
        for root, dirs, files in os.walk(directory_name, topdown=True, followlinks=self.followlinks):

            if self.hash_directory_names:
                collected_directories.extend(
                    [os.path.join(root, f) for f in dirs
                     if not self.is_excluded_directory(f)])
            collected_files.extend(
                [os.path.join(root, f) for f in files
                 if not self.is_excluded_file(f) and not self.has_excluded_extension(f)])

    def _reduce_hash(self, hashes):
        hasher = self.hash_func()
        for hashvalue in hashes:
            hasher.update(hashvalue)
        return hasher.digest()


class PathHasher:
    def __init__(self, hashfunc: str = 'md5', hash_permissions: bool = False):
        self.hash_permissions = hash_permissions
        self.hash_func = HASH_FUNCTIONS.get(hashfunc)
        if not self.hash_func:
            raise NotImplementedError('{} not implemented.'.format(hashfunc))

    def hash(self, filepath: str):
        hasher = self.hash_func()
        hasher.update(filepath.encode('utf-8'))
        if self.hash_permissions:
            stat_result = os.stat(filepath)
            # we only check the executable right of the user, because git only remembers this
            user_has_executable_rights = stat.S_IXUSR & stat_result[stat.ST_MODE]
            hasher.update(str(user_has_executable_rights).encode("utf-8"))
        return hasher.digest()


class FileContentHasher:

    def __init__(self, hashfunc: str = 'md5', blocksize: str = "64kb"):
        self.blocksize = humanfriendly.parse_size(blocksize)
        self.hash_func = HASH_FUNCTIONS.get(hashfunc)
        if not self.hash_func:
            raise NotImplementedError('{} not implemented.'.format(hashfunc))

    def hash(self, filepath: str):
        hasher = self.hash_func()
        with open(os.path.join(filepath), 'rb') as fp:
            while True:
                data = fp.read(self.blocksize)
                if not data:
                    break
                hasher.update(data)
        return hasher.digest()
