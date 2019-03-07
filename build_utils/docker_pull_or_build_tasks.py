import base64
import hashlib
import os
import shutil
import tempfile
from typing import List, Dict

import docker
import luigi
from luigi import LocalTarget

from build_utils.directory_hasher import FileDirectoryListHasher
from build_utils.docker_image_target import DockerImageTarget
from build_utils.docker_image_task import DockerImageTask


class DockerPullOrBuildImageTask(DockerImageTask):
    force_pull = luigi.BoolParameter(False)
    force_build = luigi.BoolParameter(False)
    build_directories = luigi.DictParameter()  # Format: {"relative_path_in_build_context":"source_path"}
    dockerfile = luigi.Parameter()
    log_build_context_content = luigi.BoolParameter(False)
    dont_remove_build_context = luigi.BoolParameter(False)
    build_context_base_directory = luigi.OptionalParameter(None)
    ouput_directory = luigi.Parameter("ouput")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name_target = luigi.LocalTarget("%s/image_names/%s" % (self.ouput_directory, self.image_tag))
        if self.image_name_target.exists():
            self.image_name_target.remove()
        self.log_file_target = LocalTarget(
            "%s/logs/build/%s/%s" % (self.ouput_directory, self.image_name, self.image_tag))
        if self.log_file_target.exists():
            self.log_file_target.remove()

    def output(self):
        return {"image_name_target": self.image_name_target, "log_file_target": self.log_file_target}

    def run(self):
        images_names_of_dependencies = self.get_image_names_of_dependencies()
        image_hash = self.generate_image_hash(images_names_of_dependencies)
        complete_tag = self.image_tag + "_" + image_hash
        image_target = DockerImageTarget(self.image_name, complete_tag)

        output_generator = None
        if self.force_build or self.force_pull:
            if image_target.exists():
                self._client.images.remove(image=image_target.get_complete_name())
                print(f"Removed docker images {image_target.get_complete_name()}")
        if not image_target.exists():
            if not self.force_build and self.is_image_in_registry():
                output_generator=self.pull_image(image_target)
            else:
                output_generator=self.build_image(image_target)
        self.write_output_log(output_generator)
        image_name_file = self.output()["image_name_target"]
        with image_name_file.open("wt") as log_file:
            log_file.write(image_target.get_complete_name())

    def get_image_names_of_dependencies(self):
        images_names_of_dependencies = []
        if isinstance(self.input(), List):
            for current in self.input():
                if isinstance(current, Dict) and "image_name_target" in current:
                    with current["image_name_target"].open("r") as file:
                        images_names_of_dependencies.append(file.read())
        return images_names_of_dependencies

    def pull_image(self, image_target):
        print("execute pull")
        self._client.images.pull(image_target.get_complete_name())

    def build_image(self, image_target):
        print("execute build")
        try:
            temp_directory = tempfile.mkdtemp(prefix="script_langauge_container_tmp_dir",
                                              dir=self.build_context_base_directory)
            self.copy_build_context_to_temp_dir(temp_directory)
            image, output_generator = \
                self._client.images.build(path=temp_directory,
                                          tag=image_target.get_complete_name(),
                                          rm=True)
            return output_generator
        finally:
            shutil.rmtree(temp_directory)

    def copy_build_context_to_temp_dir(self, temp_directory):
        for dest, src in self.build_directories.items():
            shutil.copytree(src, temp_directory + "/" + dest)
        shutil.copy2(self.dockerfile, temp_directory + "/Dockerfile")
        if self.log_build_context_content:
            print(self.get_files_in_build_context(temp_directory))

    def write_output_log(self, output_generator):
        log_output = self.output()["log_file_target"]
        with log_output.open("w") as log_file:
            if output_generator is not None:
                for log_line in output_generator:
                    log_file.write(str(log_line))
                    log_file.write("\n")

    def get_files_in_build_context(self, temp_directory):
        return [os.path.join(r, file) for r, d, f in os.walk(temp_directory) for file in f]

    def is_image_in_registry(self):
        try:
            registry_data = self._client.images.get_registry_data(self.get_complete_name())
            return True
        except docker.errors.APIError as e:
            print("Exception while checking if image exists in registry", self.get_complete_name(), e)
            return False

    def generate_image_hash(self, images_names_of_dependencies: List[str]):
        hash_of_build_context = self.generate_build_context_hash()
        final_hash = self.generate_final_hash(hash_of_build_context, images_names_of_dependencies)
        return self.encode_hash(final_hash)

    def generate_build_context_hash(self):
        files_directories_list_hasher = \
            FileDirectoryListHasher(hashfunc="sha256",
                                    hash_file_names=True,
                                    hash_directory_names=True,
                                    hash_permissions=True)
        files_directories_to_hash = list(self.build_directories.values()) + [str(self.dockerfile)]
        hash_of_build_context = files_directories_list_hasher.hash(files_directories_to_hash)
        return hash_of_build_context

    def generate_final_hash(self, hash_of_build_context, images_names_of_dependencies):
        hasher = hashlib.sha256()
        for image_name in sorted(images_names_of_dependencies):
            hasher.update(image_name.encode("utf-8"))
            hasher.update(hash_of_build_context)
        final_hash = hasher.digest()
        return final_hash

    def encode_hash(self, hash_of_build_directory):
        base32 = base64.b32encode(hash_of_build_directory)
        ascii = base32.decode("ASCII")
        trim = ascii.replace("=", "")
        return trim
