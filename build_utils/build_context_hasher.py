import base64
import hashlib
from typing import Dict

from build_utils.directory_hasher import FileDirectoryListHasher
from build_utils.image_info import ImageInfo


class BuildContextHasher:

    def __init__(self,build_directories_mapping:Dict[str,str], dockerfile:str):
        self.dockerfile = dockerfile
        self.build_directories_mapping = build_directories_mapping

    def generate_image_hash(self, image_info_of_dependencies: Dict[str, ImageInfo]):
        hash_of_build_context = self._generate_build_context_hash()
        final_hash = self._generate_final_hash(hash_of_build_context, image_info_of_dependencies)
        return self._encode_hash(final_hash)

    def _generate_build_context_hash(self):
        files_directories_list_hasher = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    hash_file_names=True,
                                    hash_directory_names=True,
                                    hash_permissions=True)
        files_directories_to_hash = list(self.build_directories_mapping.values()) + [str(self.dockerfile)]
        hash_of_build_context = files_directories_list_hasher.hash(files_directories_to_hash)
        return hash_of_build_context

    def _generate_final_hash(self, hash_of_build_context: bytes, image_info_of_dependencies: Dict[str, ImageInfo]):
        hashes_of_dependencies = \
            {key:image_info.hash for key, image_info in image_info_of_dependencies.items()}
        hasher = hashlib.sha256()
        for key, image_name in sorted(hashes_of_dependencies.items()):
            hasher.update(key.encode("utf-8"))
            hasher.update(image_name.encode("utf-8"))
        hasher.update(hash_of_build_context)
        final_hash = hasher.digest()
        return final_hash

    def _encode_hash(self, hash_of_build_directory):
        base32 = base64.b32encode(hash_of_build_directory)
        ascii = base32.decode("ASCII")
        trim = ascii.replace("=", "")
        return trim
