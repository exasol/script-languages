import io
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

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask
from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.data.container_info import ContainerInfo
from exaslct_src.lib.data.database_info import DatabaseInfo
from exaslct_src.lib.data.docker_network_info import DockerNetworkInfo
from exaslct_src.lib.data.image_info import ImageInfo
from exaslct_src.lib.docker.pull_log_handler import PullLogHandler
from exaslct_src.lib.docker_config import docker_client_config
from exaslct_src.lib.still_running_logger import StillRunningLogger
from exaslct_src.lib.test_runner.docker_db_test_environment_parameter import DockerDBTestEnvironmentParameter

BUCKETFS_PORT = "6583"
DB_PORT = "8888"


class SpawnTestDockerDatabase(DependencyLoggerBaseTask, DockerDBTestEnvironmentParameter):
    environment_name = luigi.Parameter()
    db_container_name = luigi.Parameter()
    network_info = JsonPickleParameter(DockerNetworkInfo, significant=False)  # type: DockerNetworkInfo
    ip_address_index_in_subnet = luigi.IntParameter(significant=False)
    attempt = luigi.IntParameter(1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = docker_client_config().get_client()
        self._low_level_client = docker_client_config().get_low_level_client()
        if self.ip_address_index_in_subnet < 0:
            raise Exception(
                "ip_address_index_in_subnet needs to be greater than 0 got %s"
                % self.ip_address_index_in_subnet)
        self.db_version = "-".join(self.docker_db_image_version.split("-")[0:-1])
        self.docker_db_config_path = f"docker_db_config/{self.db_version}"

    def __del__(self):
        self._client.close()
        self._low_level_client.close()

    def run_task(self):
        subnet = netaddr.IPNetwork(self.network_info.subnet)
        db_ip_address = str(subnet[2 + self.ip_address_index_in_subnet])
        db_private_network = "{ip}/{prefix}".format(ip=db_ip_address, prefix=subnet.prefixlen)
        database_info = None
        if self.network_info.reused:
            database_info = self._try_to_reuse_database(db_ip_address)
        if database_info is None:
            database_info = self._create_database_container(db_ip_address, db_private_network)
        self.return_object(database_info)

    def _try_to_reuse_database(self, db_ip_address: str) -> DatabaseInfo:
        self.logger.info("Try to reuse database container %s",
                          self.db_container_name)
        database_info = None
        try:
            database_info = self._create_database_info(db_ip_address)
        except Exception as e:
            self.logger.warning("Tried to reuse database container %s, but got Exeception %s. "
                                "Fallback to create new database.", self.db_container_name, e)
        return database_info

    def _handle_output(self, output_generator, image_info: ImageInfo):
        log_file_path = self.get_log_path().joinpath("pull_docker_db_image.log")
        with PullLogHandler(log_file_path, self.logger, image_info) as log_hanlder:
            still_running_logger = StillRunningLogger(
                self.logger, "pull image %s" % image_info.get_source_complete_name())
            for log_line in output_generator:
                still_running_logger.log()
                log_hanlder.handle_log_line(log_line)

    def _create_database_container(self, db_ip_address: str, db_private_network: str):
        self.logger.info("Starting database container %s", self.db_container_name)
        try:
            self._client.containers.get(self.db_container_name).remove(force=True, v=True)
        except:
            pass
        docker_db_image_info = self._pull_docker_db_images_if_necassary()
        db_volume = self._prepare_db_volume(db_private_network, docker_db_image_info)
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
        docker_network = self._client.networks.get(self.network_info.network_name)
        network_aliases = self._get_network_aliases()
        docker_network.connect(db_container, ipv4_address=db_ip_address, aliases=network_aliases)
        db_container.start()
        database_info = self._create_database_info(db_ip_address)
        return database_info

    def _get_network_aliases(self):
        network_aliases = ["exasol_test_database", self.db_container_name]
        return network_aliases

    def _create_database_info(self, db_ip_address: str) -> DatabaseInfo:
        db_container = self._client.containers.get(self.db_container_name)
        if db_container.status != "running":
            raise Exception(f"Container {self.db_container_name} not running")
        network_aliases = self._get_network_aliases()
        container_info = \
            ContainerInfo(container_name=self.db_container_name,
                          ip_address=db_ip_address,
                          network_aliases=network_aliases,
                          network_info=self.network_info,
                          volume_name=self._get_db_volume_name())
        database_info = \
            DatabaseInfo(host=db_ip_address, db_port=DB_PORT, bucketfs_port=BUCKETFS_PORT,
                         container_info=container_info)
        return database_info

    def _pull_docker_db_images_if_necassary(self):
        image_name = "exasol/docker-db"
        docker_db_image_info = ImageInfo(
            target_repository_name=image_name,
            source_repository_name=image_name,
            source_tag=self.docker_db_image_version,
            target_tag=self.docker_db_image_version,
            hash="", commit="",
            image_description=None)
        try:
            self._client.images.get(docker_db_image_info.get_source_complete_name())
        except docker.errors.ImageNotFound as e:
            self.logger.info("Pulling docker-db image %s",
                             docker_db_image_info.get_source_complete_name())
            output_generator = self._low_level_client.pull(
                docker_db_image_info.source_repository_name,
                tag=docker_db_image_info.source_tag,
                stream=True)
            self._handle_output(output_generator, docker_db_image_info)
        return docker_db_image_info

    def _prepare_db_volume(self, db_private_network: str,
                           docker_db_image_info: ImageInfo) -> Volume:
        db_volume_preperation_container_name = f"""{self.db_container_name}_preparation"""
        db_volume_name = self._get_db_volume_name()
        self._remove_container(db_volume_preperation_container_name)
        self._remove_volume(db_volume_name)
        db_volume, volume_preparation_container = \
            volume_preparation_container, volume_preparation_container = \
            self._create_volume_and_container(db_volume_name,
                                              db_volume_preperation_container_name)
        try:
            self._upload_init_db_files(volume_preparation_container,
                                       db_private_network,
                                       docker_db_image_info)
            self._execute_init_db(db_volume, volume_preparation_container)
            return db_volume
        finally:
            volume_preparation_container.remove(force=True)

    def _get_db_volume_name(self):
        db_volume_name = f"""{self.db_container_name}_volume"""
        return db_volume_name

    def _remove_container(self, db_volume_preperation_container_name):
        try:
            self._client.containers.get(db_volume_preperation_container_name).remove(force=True)
            self.logger.info("Removed container %s", db_volume_preperation_container_name)
        except docker.errors.NotFound:
            pass

    def _remove_volume(self, db_volume_name):
        try:
            self._client.volumes.get(db_volume_name).remove(force=True)
            self.logger.info("Removed volume %s", db_volume_name)
        except docker.errors.NotFound:
            pass

    def _create_volume_and_container(self, db_volume_name, db_volume_preperation_container_name) \
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

    def _upload_init_db_files(self,
                              volume_preperation_container: Container,
                              db_private_network: str,
                              docker_db_image_info: ImageInfo):
        file_like_object = io.BytesIO()
        with tarfile.open(fileobj=file_like_object, mode="x") as tar:
            tar.add(f"{self.docker_db_config_path}/init_db.sh", "init_db.sh")
            self._add_exa_conf(tar, db_private_network, docker_db_image_info)
        volume_preperation_container.put_archive("/", file_like_object.getbuffer().tobytes())

    def _add_exa_conf(self, tar: tarfile.TarFile,
                      db_private_network: str,
                      docker_db_image_info: ImageInfo):
        with open(f"{self.docker_db_config_path}/EXAConf") as file:
            template_str = file.read()
        template = Template(template_str)
        rendered_template = template.render(private_network=db_private_network,
                                            db_version=self.db_version,
                                            image_version=self.docker_db_image_version)
        self._add_string_to_tarfile(tar, "EXAConf", rendered_template)

    def _add_string_to_tarfile(self, tar: tarfile.TarFile, name: str, string: str):
        encoded = string.encode('utf-8')
        bytes_io = io.BytesIO(encoded)
        tar_info = tarfile.TarInfo(name=name)
        tar_info.mtime = time.time()
        tar_info.size = len(encoded)
        tar.addfile(tarinfo=tar_info, fileobj=bytes_io)

    def _execute_init_db(self, db_volume: Volume, volume_preperation_container: Container):
        (exit_code, output) = volume_preperation_container.exec_run(cmd="bash /init_db.sh")
        if exit_code != 0:
            raise Exception(
                "Error during preperation of docker-db volume %s got following output %s" % (db_volume.name, output))
