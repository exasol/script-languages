import logging
import os
import pathlib
import shutil
import subprocess
import tempfile
from datetime import datetime

import docker
import humanfriendly
import luigi

from build_utils.lib.abstract_log_handler import AbstractLogHandler
from build_utils.lib.build_config import build_config
from build_utils.lib.data.dependency_collector.dependency_image_info_collector import DependencyImageInfoCollector
from build_utils.lib.data.dependency_collector.dependency_release_info_collector import RELEASE_INFO
from build_utils.lib.data.release_info import ReleaseInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.flavor import flavor
from build_utils.lib.log_config import WriteLogFilesToConsole
from build_utils.lib.still_running_logger import StillRunningLogger
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
               datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
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
                path=str(release_file),
                complete_name=release_name,
                name=flavor.get_name_from_path(self.flavor_path),
                hash=image_info_of_release_image.hash,
                is_new=is_new,
                depends_on_image=image_info_of_release_image,
                release_type=self.get_release_type())
            file.write(release_info.to_json())

    def create_release(self, release_image_name: str, release_file: str):
        self.logger.info("Task %s: Create release file %s", self.task_id, release_file)
        temp_directory = tempfile.mkdtemp(prefix="release_archive",
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
        self.logger.info("Task %s: Export release container %s", self.task_id, release_image_name)
        client = docker.DockerClient(base_url=self._docker_config.base_url)
        try:
            container = client.containers.create(image=release_image_name)
            try:
                return self.export_container(container, release_image_name, temp_directory)
            finally:
                container.remove(force=True)
        finally:
            client.close()

    def export_container(self, container, release_image_name:str, temp_directory:str):
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
        self.logger.info("Task %s: Pack release file %s", self.task_id, release_file)
        extract_content = " ".join(os.listdir(extract_dir))
        command = f"""tar -C {extract_dir} -cvzf {release_file} {extract_content}"""
        self.run_command(command, "packing release file %s" % release_file,
                         log_path.joinpath("pack_release_file.log"))

    def modify_extracted_container(self, extract_dir: str):
        os.symlink("/conf/resolv.conf", f"""{extract_dir}/etc/resolv.conf""")
        os.symlink("/conf/hosts", f"""{extract_dir}/etc/hosts""")

    def extract_exported_container(self, log_path: pathlib.Path, export_file: str, temp_directory: str):
        self.logger.info("Task %s: Extract exported file %s", self.task_id, export_file)
        extract_dir = temp_directory + "/extract"
        os.makedirs(extract_dir)
        excludes = " ".join(
            ["--exclude=%s" % dir for dir in ["dev/*", "proc/*", "etc/resolv.conf", "etc/hosts"]])
        self.run_command(f"""tar {excludes} -xvf {export_file} -C {extract_dir}""",
                         "extracting exported container %s" % export_file,
                         log_path.joinpath("extract_release_file.log"))
        return extract_dir

    def prepare_log_dir(self, release_image_name: str):
        log_dir = pathlib.Path("%s/logs/release/%s/%s/"
                               % (self._build_config.ouput_directory,
                                  release_image_name,
                                  datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        if log_dir.exists():
            shutil.rmtree(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    def run_command(self, command: str, description: str, log_file_path: pathlib.Path):
        with subprocess.Popen(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            with CommandLogHandler(log_file_path, self.logger, self.task_id, description) as log_handler:
                still_running_logger = StillRunningLogger(
                    self.logger, self.task_id, description)
                log_handler.handle_log_line(command.encode("utf-8"))
                for line in iter(process.stdout.readline, b''):
                    still_running_logger.log()
                    log_handler.handle_log_line(line)
                process.wait(timeout=60*2)
                return_code_log_line = "return code %s" % process.returncode
                log_handler.handle_log_line(return_code_log_line.encode("utf-8"), process.returncode != 0)


class CommandLogHandler(AbstractLogHandler):

    def __init__(self, log_file_path: pathlib.Path, logger, task_id, description: str):
        super().__init__(log_file_path, logger, task_id)
        self._description = description

    def handle_log_line(self, log_line, error: bool = False):
        log_line = log_line.decode("utf-8")
        self._log_file.write(log_line)
        self._complete_log.append(log_line)

    def finish(self):
        if self._log_config.write_log_files_to_console==WriteLogFilesToConsole.all:
            self._logger.info("Task %s: Command log for %s \n%s",
                              self._task_id,
                              self._description,
                              "".join(self._complete_log))
        if self._error_message is not None:
            if self._log_config.write_log_files_to_console == WriteLogFilesToConsole.only_error:
                self._logger.error("Task %s: Command failed %s failed\nCommand Log:\n%s",
                                  self._task_id,
                                  self._description,
                                  "\n".join(self._complete_log))
            raise Exception(
                "Error occured during %s. Received error \"%s\" ."
                "The whole log can be found in %s"
                % (self._description,
                   self._error_message,
                   self._log_file_path.absolute()),
                self._log_file_path.absolute())
