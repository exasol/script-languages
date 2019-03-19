import datetime
import logging
import os
import pathlib
import shutil
import subprocess
import tempfile

import docker
import humanfriendly
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.lib.data.release_info import ReleaseInfo
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import RELEASE_INFO
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.release_type import ReleaseType


class ReleaseContainerTask(luigi.Task):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/releases/%s/%s/%s"
            % (self._build_config.ouput_directory,
               flavor.get_name_from_path(self.flavor_path),
               self.get_release_type().name,
               datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return {RELEASE_INFO: self._target}

    def requires(self):
        return self.get_release_task(self.flavor_path)

    def get_release_task(self, flavor_path):
        pass

    def get_release_type(self) -> ReleaseType:
        pass

    def run(self):
        image_info_of_release_image = DependencyImageInfoCollector().get_from_sinlge_input(self.input())
        release_image_name = image_info_of_release_image.complete_name
        release_path = pathlib.Path(self._build_config.ouput_directory).joinpath("releases")
        release_path.mkdir(parents=True, exist_ok=True)
        release_name = f"""{image_info_of_release_image.tag}-{image_info_of_release_image.hash}"""
        release_file = release_path.joinpath(release_name + ".tar.gz")
        if release_file.exists() and \
                (self._build_config.force_build or
                 self._build_config.force_pull):
            self.logger.info("Task %s: Removed release file %s", self.task_id, release_file)
            os.remove(release_file)

        is_new = False
        if not release_file.exists():
            self.create_release(release_image_name, release_file)
            is_new = True

        with self.output()[RELEASE_INFO].open("w") as file:
            release_info = ReleaseInfo(
                path=release_file,
                complete_name=release_name,
                name=image_info_of_release_image.tag,
                hash=image_info_of_release_image.hash,
                is_new=is_new,
                depends_on_image=image_info_of_release_image,
                release_type=self.get_release_type())
            file.write(release_info.to_json())

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
