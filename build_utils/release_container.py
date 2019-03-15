import json
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile

import docker
import humanfriendly
import luigi

from build_utils.build_config import build_config
from build_utils.docker_build import DockerBuild_Release
from build_utils.docker_config import docker_config
from build_utils.docker_pull_or_build_flavor_image_task import flavor
from build_utils.image_dependency_collector import ImageDependencyCollector
from build_utils.image_info import ImageInfo


class ReleaseContainer(luigi.Task):
    logger = logging.getLogger('luigi-interface')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._flavor_config = flavor()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/release"
            % (self._build_config.ouput_directory))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return self._target

    def requires(self):
        return {"release": DockerBuild_Release()}

    def run(self):
        image_info_of_dependencies = ImageDependencyCollector().get_dict_of_image_info_of_dependencies(self.input())
        with self.output().open("w") as file:
            json.dump(ImageInfo.merge_dependencies(image_info_of_dependencies), file)
        image_info_of_release_image = image_info_of_dependencies["release"]
        release_image_name = image_info_of_release_image.complete_name
        release_path = pathlib.Path(self._build_config.ouput_directory).joinpath("releases")
        release_path.mkdir(parents=True, exist_ok=True)
        release_file = \
            release_path.joinpath(
                f"""{self._flavor_config.get_flavor_name()}-{image_info_of_release_image.hash}.tar.gz""")
        if release_file.exists() and \
                (self._build_config.force_build or
                 self._build_config.force_pull):
            self.logger.info("Task %s: Removed release file %s", self.task_id, release_file)
            os.remove(release_file)

        if not release_file.exists():
            self.create_release(release_image_name, release_file)

    def create_release(self, release_image_name, release_file):
        self.logger.info("Task %s: Create release file %s", self.task_id, release_file)
        temp_directory = tempfile.mkdtemp(prefix="release_archive",
                                          dir=self._build_config.temporary_base_directory)
        try:
            export_file = self.export_container(release_image_name, temp_directory)
            extract_dir = self.extract_exported_container(export_file, temp_directory)
            self.modify_extracted_container(extract_dir)
            self.create_release_file(extract_dir, release_file)
        finally:
            shutil.rmtree(temp_directory)

    def export_container(self, release_image_name, temp_directory):
        client = docker.DockerClient(base_url=self._docker_config.base_url)
        try:
            container = client.containers.create(image=release_image_name)
            try:
                generator = container.export(chunk_size=humanfriendly.parse_size("100mb"))
                export_file = temp_directory + "/export.tar"
                with open(export_file, "wb") as file:
                    for chunk in generator:
                        file.write(chunk)
                return export_file
            finally:
                container.remove(force=True)
        finally:
            client.close()

    def create_release_file(self, extract_dir, release_file):
        extract_content = " ".join(os.listdir(extract_dir))
        self.run_tar(f"""tar -C {extract_dir} -czf {release_file} {extract_content}""")

    def modify_extracted_container(self, extract_dir):
        os.symlink("/conf/resolv.conf", f"""{extract_dir}/etc/resolv.conf""")
        os.symlink("/conf/hosts", f"""{extract_dir}/etc/hosts""")

    def extract_exported_container(self, export_file, temp_directory):
        extract_dir = temp_directory + "/extract"
        os.makedirs(extract_dir)
        excludes = " ".join(
            ["--exclude=%s" % dir for dir in ["dev/*", "proc/*", "etc/resolv.conf", "etc/hosts"]])
        self.run_tar(f"""tar {excludes} -xf {export_file} -C {extract_dir}""")
        return extract_dir

    def run_tar(self, tar):
        tar_status = subprocess.run(tar.split(" "))
        tar_status.check_returncode()
