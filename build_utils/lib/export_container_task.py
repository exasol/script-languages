import logging
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
from datetime import datetime

import docker
import humanfriendly
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.command_log_handler import CommandLogHandler
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import RELEASE_INFO
from build_utils.lib.data.image_info import ImageInfo
from build_utils.lib.data.release_info import ExportInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.lib.still_running_logger import StillRunningLogger
from build_utils.lib.test_runner.create_export_directory import CreateExportDirectory
from build_utils.stoppable_task import StoppableTask
from build_utils.release_type import ReleaseType


class ExportContainerTask(StoppableTask):
    logger = logging.getLogger('luigi-interface')
    flavor_path = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._prepare_outputs()

    def _prepare_outputs(self):
        self._target = luigi.LocalTarget(
            "%s/info/export/%s/%s"
            % (self._build_config.output_directory,
               flavor.get_name_from_path(self.flavor_path),
               self.get_release_type().name
               # datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
               ))
        if self._target.exists():
            self._target.remove()

    def output(self):
        return {RELEASE_INFO: self._target}

    def requires(self):
        return {"release_task": self.get_release_task(self.flavor_path),
                "export_directory": CreateExportDirectory()}

    def get_release_task(self, flavor_path):
        pass

    def get_release_type(self) -> ReleaseType:
        pass

    def run_task(self):
        image_infos = DependencyImageInfoCollector().get_from_dict_of_inputs(self.input())
        image_info_of_release_image = image_infos["release_task"]
        release_image_name = image_info_of_release_image.complete_name
        release_path = pathlib.Path(self.get_release_directory()).absolute()
        release_name = f"""{image_info_of_release_image.tag}-{image_info_of_release_image.hash}"""
        release_file = release_path.joinpath(release_name + ".tar.gz").absolute()
        self.remove_release_file_if_requested(release_file)

        is_new = False
        if not release_file.exists():
            self.create_release(release_image_name, release_file)
            is_new = True

        self.write_release_info(image_info_of_release_image, is_new, release_file, release_name)

    def get_release_directory(self):
        return pathlib.Path(self.input()["export_directory"].path).absolute().parent

    def remove_release_file_if_requested(self, release_file):
        if release_file.exists() and \
                (self._build_config.force_build or
                 self._build_config.force_pull):
            self.logger.info("Task %s: Removed container file %s", self.task_id, release_file)
            os.remove(release_file)

    def write_release_info(self, image_info_of_release_image: ImageInfo, is_new: bool,
                           release_file: pathlib.Path, release_name: str):
        release_info = ExportInfo(
            path=str(release_file),
            complete_name=release_name,
            name=flavor.get_name_from_path(self.flavor_path),
            hash=image_info_of_release_image.hash,
            is_new=is_new,
            depends_on_image=image_info_of_release_image,
            release_type=self.get_release_type())
        json = release_info.to_json()
        with self.output()[RELEASE_INFO].open("w") as file:
            file.write(json)

    def create_release(self, release_image_name: str, release_file: str):
        self.logger.info("Task %s: Create container file %s", self.task_id, release_file)
        temp_directory = tempfile.mkdtemp(prefix="release_archive_",
                                          dir=self._build_config.temporary_base_directory)
        try:
            log_path = self.prepare_log_dir(release_image_name)
            export_file = self.create_and_export_container(release_image_name, temp_directory)
            extract_dir = self.extract_exported_container(log_path, export_file, temp_directory)
            self.modify_extracted_container(extract_dir)
            self.pack_release_file(log_path, extract_dir, release_file)
        finally:
            shutil.rmtree(temp_directory)

    def create_and_export_container(self, release_image_name: str, temp_directory: str):
        self.logger.info("Task %s: Export container %s", self.task_id, release_image_name)
        client = docker.DockerClient(base_url=self._docker_config.base_url)
        try:
            container = client.containers.create(image=release_image_name)
            try:
                return self.export_container(container, release_image_name, temp_directory)
            finally:
                container.remove(force=True)
        finally:
            client.close()

    def export_container(self, container, release_image_name: str, temp_directory: str):
        generator = container.export(chunk_size=humanfriendly.parse_size("10mb"))
        export_file = temp_directory + "/export.tar"
        with open(export_file, "wb") as file:
            still_running_logger = StillRunningLogger(
                self.logger, self.task_id, "Export image %s" % release_image_name)
            for chunk in generator:
                still_running_logger.log()
                file.write(chunk)
        return export_file

    def pack_release_file(self, log_path: pathlib.Path, extract_dir: str, release_file: str):
        self.logger.info("Task %s: Pack container file %s", self.task_id, release_file)
        extract_content = " ".join("'%s'" % file for file in os.listdir(extract_dir))
        command = f"""tar -C '{extract_dir}' -cvzf '{release_file}' {extract_content}"""
        self.run_command(command, "packing container file %s" % release_file,
                         log_path.joinpath("pack_release_file.log"))

    def modify_extracted_container(self, extract_dir: str):
        os.symlink("/conf/resolv.conf", f"""{extract_dir}/etc/resolv.conf""")
        os.symlink("/conf/hosts", f"""{extract_dir}/etc/hosts""")

    def extract_exported_container(self, log_path: pathlib.Path, export_file: str, temp_directory: str):
        self.logger.info("Task %s: Extract exported file %s", self.task_id, export_file)
        extract_dir = temp_directory + "/extract"
        os.makedirs(extract_dir)
        excludes = " ".join(
            ["--exclude='%s'" % dir for dir in ["dev/*", "proc/*", "etc/resolv.conf", "etc/hosts"]])
        self.run_command(f"""tar {excludes} -xvf '{export_file}' -C '{extract_dir}'""",
                         "extracting exported container %s" % export_file,
                         log_path.joinpath("extract_release_file.log"))
        return extract_dir

    def prepare_log_dir(self, release_image_name: str):
        log_dir = pathlib.Path("%s/logs/exports/%s/%s/"
                               % (self._build_config.output_directory,
                                  release_image_name,
                                  datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def run_command(self, command: str, description: str, log_file_path: pathlib.Path):
        with subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            with CommandLogHandler(log_file_path, self.logger, self.task_id, description) as log_handler:
                still_running_logger = StillRunningLogger(
                    self.logger, self.task_id, description)
                log_handler.handle_log_line((command + "\n").encode("utf-8"))
                for line in iter(process.stdout.readline, b''):
                    still_running_logger.log()
                    log_handler.handle_log_line(line)
                process.wait(timeout=60 * 2)
                return_code_log_line = "return code %s" % process.returncode
                log_handler.handle_log_line(return_code_log_line.encode("utf-8"), process.returncode != 0)
