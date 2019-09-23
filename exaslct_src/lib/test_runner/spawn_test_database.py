import io
import logging
import pathlib
import tarfile
import time
from datetime import datetime
from typing import Tuple

import docker
import luigi
import netaddr
from docker.models.containers import Container
from docker.models.volumes import Volume
from jinja2 import Template

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.dependency_collector.dependency_database_info_collector import DATABASE_INFO
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker.pull_log_handler import PullLogHandler
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.stoppable_task import StoppableTask

BUCKETFS_PORT = "6583"
DB_PORT = "8888"


class SpawnTestDockerDatabase(StoppableTask):
    logger = logging.getLogger('luigi-interface')

    environment_name = luigi.Parameter()
    db_container_name = luigi.Parameter()
    docker_db_image_version = luigi.OptionalParameter("6.2.1-d1")
    docker_db_image_name = luigi.OptionalParameter("exasol/docker-db")
    reuse_database = luigi.BoolParameter(False, significant=False)
    network_info_dict = luigi.DictParameter(significant=False)
    ip_address_index_in_subnet = luigi.IntParameter(significant=False)
    database_port_forward = luigi.OptionalParameter(None, significant=False)
    bucketfs_port_forward = luigi.OptionalParameter(None, significant=False)
    attempt = luigi.IntParameter(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()
        if self.ip_address_index_in_subnet < 0:
            raise Exception(
                "ip_address_index_in_subnet needs to be greater than 0 got %s"
                % self.ip_address_index_in_subnet)
        self._prepare_outputs()
        self.db_version = "-".join(self.docker_db_image_version.split("-")[0:-1])
        self.docker_db_config_path = f"docker_db_config/{self.db_version}"

    def __del__(self):
        self._client.close()
        self._low_level_client.close()

    def _prepare_outputs(self):
        self._database_info_target = luigi.LocalTarget(
            "%s/info/environment/%s/database/%s/%s/database_info"
            % (build_config().output_directory,
               self.environment_name,
               self.db_container_name,
               self.attempt))
        if self._database_info_target.exists():
            self._database_info_target.remove()

    def output(self):
        return {DATABASE_INFO: self._database_info_target}

    def run_task(self):
        network_info = DockerNetworkInfo.from_dict(self.network_info_dict)
        subnet = netaddr.IPNetwork(network_info.subnet)
        db_ip_address = str(subnet[2 + self.ip_address_index_in_subnet])
        db_private_network = "{ip}/{prefix}".format(ip=db_ip_address, prefix=subnet.prefixlen)
        database_info = None
        if network_info.reused:
            database_info = self.try_to_reuse_database(db_ip_address, network_info)
        if database_info is None:
            database_info = self.create_database_container(db_ip_address, db_private_network, network_info)
        self.write_output(database_info)

    def try_to_reuse_database(self, db_ip_address: str, network_info: DockerNetworkInfo) -> DatabaseInfo:
        self.logger.info("Task %s: Try to reuse database container %s",
                         self.__repr__(), self.db_container_name)
        database_info = None
        try:
            database_info = self.get_database_info(db_ip_address, network_info)
        except Exception as e:
            self.logger.warning("Task %s: Tried to reuse database container %s, but got Exeception %s. "
                                "Fallback to create new database.", self.__repr__(), self.db_container_name, e)
        return database_info

    def write_output(self, database_info: DatabaseInfo):
        with self.output()[DATABASE_INFO].open("w") as file:
            file.write(database_info.to_json())

    def get_database_info(self, db_ip_address: str,
                          network_info: DockerNetworkInfo) -> DatabaseInfo:
        db_container = self._client.containers.get(self.db_container_name)
        if db_container.status != "running":
            raise Exception(f"Container {self.db_container_name} not running")
        container_info = \
            ContainerInfo(container_name=self.db_container_name,
                          ip_address=db_ip_address,
                          network_info=network_info,
                          volume_name=self.get_db_volume_name())
        database_info = \
            DatabaseInfo(host=db_ip_address, db_port=DB_PORT, bucketfs_port=BUCKETFS_PORT,
                         container_info=container_info)
        return database_info

    def _handle_output(self, output_generator, image_info: ImageInfo):
        log_file_path = self.prepate_log_file_path(image_info)
        with PullLogHandler(log_file_path, self.logger, self.__repr__(), image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, self.__repr__(), "pull image %s" % image_info.get_source_complete_name())
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)

    def prepate_log_file_path(self, image_info: ImageInfo):
        log_file_path = pathlib.Path("%s/logs/docker-pull/%s/%s/%s"
                                     % (build_config().output_directory,
                                        image_info.source_repository_name, image_info.source_tag,
                                        datetime.now().strftime('%Y_%m_%d_%H_%M_%S')))
        log_file_path = luigi.LocalTarget(str(log_file_path))
        if log_file_path.exists():
            log_file_path.remove()
        return log_file_path

    def create_database_container(self,
                                  db_ip_address: str, db_private_network: str,
                                  network_info: DockerNetworkInfo):
        self.logger.info("Task %s: Starting database container %s",
                         self.__repr__(), self.db_container_name)
        try:
            self._client.containers.get(self.db_container_name).remove(force=True, v=True)
        except:
            pass
        docker_db_image_info = self.pull_docker_db_images_if_necassary()
        db_volume = self.prepare_db_volume(db_private_network, docker_db_image_info)
        ports = {}
        if self.database_port_forward is not None:
            ports[f"{DB_PORT}/tcp"] = ('0.0.0.0', int(self.database_port_forward))
        if self.bucketfs_port_forward is not None:
            ports[f"{BUCKETFS_PORT}/tcp"] = ('0.0.0.0', int(self.bucketfs_port_forward))
        db_container = \
            self._client.containers.create(
                image="%s" % (docker_db_image_info.get_source_complete_name()),
                name=self.db_container_name,
                detach=True,
                privileged=True,
                volumes={db_volume.name: {"bind": "/exa", "mode": "rw"}},
                network_mode=None,
                ports=ports
            )
        self._client.networks.get(network_info.network_name).connect(db_container, ipv4_address=db_ip_address)
        db_container.start()
        container_info = \
            ContainerInfo(container_name=db_container.name,
                          ip_address=db_ip_address,
                          network_info=network_info,
                          volume_name=db_volume.name)
        database_info = \
            DatabaseInfo(host=db_ip_address, db_port=DB_PORT, bucketfs_port=BUCKETFS_PORT,
                         container_info=container_info)
        return database_info

    def pull_docker_db_images_if_necassary(self):
        image_name = "exasol/docker-db"
        docker_db_image_info = ImageInfo(
            target_repository_name=image_name,
            source_repository_name=image_name,
            source_tag=self.docker_db_image_version,
            target_tag=self.docker_db_image_version,
            hash="", commit = "",
            image_description=None)
        try:

            self._client.images.get(docker_db_image_info.get_source_complete_name())
        except docker.errors.ImageNotFound as e:
            self.logger.info("Task %s: Pulling docker-db image %s",
                             self.__repr__(), docker_db_image_info.get_source_complete_name())
            output_generator = self._low_level_client.pull(docker_db_image_info.source_repository_name,
                                                           tag=docker_db_image_info.source_tag,
                                                           stream=True)
            self._handle_output(output_generator, docker_db_image_info)
        return docker_db_image_info

    def prepare_db_volume(self, db_private_network: str, docker_db_image_info: ImageInfo) -> Volume:
        db_volume_preperation_container_name = f"""{self.db_container_name}_preparation"""
        db_volume_name = self.get_db_volume_name()
        self.remove_container(db_volume_preperation_container_name)
        self.remove_volume(db_volume_name)
        db_volume, volume_preparation_container = \
            volume_preparation_container, volume_preparation_container = \
            self.create_volume_and_container(db_volume_name,
                                             db_volume_preperation_container_name)
        try:
            self.upload_init_db_files(volume_preparation_container,
                                      db_private_network,
                                      docker_db_image_info)
            self.execute_init_db(db_volume, volume_preparation_container)
            return db_volume
        finally:
            volume_preparation_container.remove(force=True)

    def get_db_volume_name(self):
        db_volume_name = f"""{self.db_container_name}_volume"""
        return db_volume_name

    def remove_container(self, db_volume_preperation_container_name):
        try:
            self._client.containers.get(db_volume_preperation_container_name).remove(force=True)
            self.logger.info("Task %s: Removed container %s", self.__repr__(), db_volume_preperation_container_name)
        except docker.errors.NotFound:
            pass

    def remove_volume(self, db_volume_name):
        try:
            self._client.volumes.get(db_volume_name).remove(force=True)
            self.logger.info("Task %s: Removed volume %s", self.__repr__(), db_volume_name)
        except docker.errors.NotFound:
            pass

    def create_volume_and_container(self, db_volume_name, db_volume_preperation_container_name) \
            -> Tuple[Volume, Container]:
        db_volume = self._client.volumes.create(db_volume_name)
        volume_preparation_container = \
            self._client.containers.run(
                image="ubuntu:18.04",
                name=db_volume_preperation_container_name,
                auto_remove=True,
                command="sleep infinity",
                detach=True,
                volumes={
                    db_volume.name: {"bind": "/exa", "mode": "rw"}})
        return db_volume, volume_preparation_container

    def upload_init_db_files(self,
                             volume_preperation_container: Container,
                             db_private_network: str,
                             docker_db_image_info: ImageInfo):
        file_like_object = io.BytesIO()
        with tarfile.open(fileobj=file_like_object, mode="x") as tar:
            tar.add(f"{self.docker_db_config_path}/init_db.sh", "init_db.sh")
            self.add_exa_conf(tar, db_private_network, docker_db_image_info)
        volume_preperation_container.put_archive("/", file_like_object.getbuffer().tobytes())

    def add_exa_conf(self, tar: tarfile.TarFile,
                     db_private_network: str,
                     docker_db_image_info: ImageInfo):
        with open(f"{self.docker_db_config_path}/EXAConf") as file:
            template_str = file.read()
        template = Template(template_str)
        rendered_template = template.render(private_network=db_private_network,
                                            db_version=self.db_version,
                                            image_version=self.docker_db_image_version)
        self.add_string_to_tarfile(tar, "EXAConf", rendered_template)

    def add_string_to_tarfile(self, tar: tarfile.TarFile, name: str, string: str):
        encoded = string.encode('utf-8')
        bytes_io = io.BytesIO(encoded)
        tar_info = tarfile.TarInfo(name=name)
        tar_info.mtime = time.time()
        tar_info.size = len(encoded)
        tar.addfile(tarinfo=tar_info, fileobj=bytes_io)

    def execute_init_db(self, db_volume: Volume, volume_preperation_container: Container):
        (exit_code, output) = volume_preperation_container.exec_run(cmd="bash /init_db.sh")
        if exit_code != 0:
            raise Exception(
                "Error during preperation of docker-db volume %s got following output %s" % (db_volume.name, output))
