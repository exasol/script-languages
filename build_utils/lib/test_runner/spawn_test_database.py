import gzip
import io
import json
import logging
import os
import pathlib
import shutil
import tarfile
import time
from typing import Tuple

import docker
import luigi
import netaddr
from docker.models.containers import Container
from docker.models.networks import Network
from docker.models.volumes import Volume
from jinja2 import Template
from netaddr import IPNetwork

from build_utils.lib.build_config import build_config
from build_utils.lib.data.container_info import ContainerInfo
from build_utils.lib.data.database_info import DatabaseInfo
from build_utils.lib.data.dependency_collector.dependency_database_info_collector import DATABASE_INFO
from build_utils.lib.data.docker_network_info import DockerNetworkInfo
from build_utils.lib.docker_config import docker_config
from build_utils.lib.test_runner.container_log_thread import ContainerLogThread

BUCKETFS_PORT = "6583"
DB_PORT = "8888"


class SpawnTestDockerDatabase(luigi.Task):
    logger = logging.getLogger('luigi-interface')
    reuse_database = luigi.BoolParameter(False)
    db_container_name = luigi.Parameter()
    db_startup_timeout_in_seconds = luigi.IntParameter(60 * 2, significant=False)
    remove_container_after_startup_failure = luigi.BoolParameter(True, significant=False)
    network_info_dict = luigi.DictParameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._low_level_client = docker.APIClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._database_info_target = luigi.LocalTarget(
            "%s/test-runner/db-test/database/%s/info"
            % (self._build_config.ouput_directory,
               self.db_container_name))
        if self._database_info_target.exists():
            self._database_info_target.remove()

    def output(self):
        return {DATABASE_INFO: self._database_info_target}

    def run(self):
        network_info = DockerNetworkInfo.from_dict(self.network_info_dict)
        subnet = netaddr.IPNetwork(network_info.subnet)
        db_ip_address = str(subnet[2])
        db_private_network = "{ip}/{prefix}".format(ip=db_ip_address, prefix=subnet.prefixlen)
        database_info = None
        if network_info.reused:
            try:
                database_info = self.get_database_info(db_ip_address, network_info)
            except Exception as e:
                self.logger.warning("Tried to reuse database container %s, but got Exeception %s. "
                                    "Fallback to create new database.", self.db_container_name, e)
        if database_info is None:
            database_info = self.create_database_container(db_ip_address, db_private_network, network_info)
        self.write_output(database_info)

    def write_output(self, database_info):
        with self.output()[DATABASE_INFO].open("w") as file:
            file.write(database_info.to_json())

    def get_database_info(self, db_ip_address: str,
                               network_info: DockerNetworkInfo):
        self._client.containers.get(self.db_container_name)
        container_info = \
            ContainerInfo(self.db_container_name, network_info=network_info,
                          volume_name=self.get_db_volume_name())
        database_info = \
            DatabaseInfo(host=db_ip_address, db_port=DB_PORT, bucketfs_port=BUCKETFS_PORT,
                         container_info=container_info)
        return database_info

    def create_database_container(self, db_ip_address: str, db_private_network: str,
                                  network_info: DockerNetworkInfo):
        db_volume = self.prepare_db_volume(db_private_network)
        db_container = \
            self._client.containers.run(
                image="exasol/docker-db:6.0.12-d1",
                name=self.db_container_name,
                detach=True,
                privileged=True,
                volumes={db_volume.name: {"bind": "/exa", "mode": "rw"}},
                network=network_info.network_name)
        database_log_path = \
            pathlib.Path("%s/test-runner/db-test/database/%s/logs/"
                         % (self._build_config.ouput_directory,
                            self.db_container_name))
        is_database_ready = self.wait_for_database_startup(database_log_path, db_container)
        after_startup_db_log_file = database_log_path.joinpath("after_startup_db_log.tar.gz")
        self.save_db_log_files_as_gzip_tar(after_startup_db_log_file, db_container)
        if is_database_ready:
            container_info = \
                ContainerInfo(db_container.name, network_info=network_info,
                              volume_name=db_volume.name)
            database_info = \
                DatabaseInfo(host=db_ip_address, db_port=DB_PORT, bucketfs_port=BUCKETFS_PORT,
                             container_info=container_info)
            return database_info
        else:
            if self.remove_container_after_startup_failure:
                db_container.remove(v=True, force=True)
            raise Exception("Startup of database in container %s failed" % db_container.name)

    def wait_for_database_startup(self, database_log_path, db_container):
        if database_log_path.exists():
            shutil.rmtree(database_log_path)
        database_log_path.mkdir(parents=True)
        startup_log_file = database_log_path.joinpath("startup.log")
        thread = ContainerLogThread(db_container,
                                    startup_log_file)
        thread.start()
        is_database_ready = self.is_database_ready(db_container, self.db_startup_timeout_in_seconds)
        thread.stop()
        thread.join()
        return is_database_ready

    def prepare_db_volume(self, db_private_network: str) -> Volume:
        db_volume_preperation_container_name = f"""{self.db_container_name}_preparation"""
        db_volume_name = self.get_db_volume_name()
        self.remove_container(db_volume_preperation_container_name)
        self.remove_volume(db_volume_name)
        db_volume, volume_preperation_container = \
            volume_preperation_container, volume_preperation_container = \
            self.create_volume_and_container(db_volume_name,
                                             db_volume_preperation_container_name)
        try:
            self.upload_init_db_files(volume_preperation_container, db_private_network)
            self.execute_init_db(db_volume, volume_preperation_container)
            return db_volume
        finally:
            volume_preperation_container.remove(force=True)

    def get_db_volume_name(self):
        db_volume_name = f"""{self.db_container_name}_volume"""
        return db_volume_name

    def remove_container(self, db_volume_preperation_container_name):
        try:
            self._client.containers.get(db_volume_preperation_container_name).remove(force=True)
            self.logger.info("Removed container %s" % db_volume_preperation_container_name)
        except docker.errors.NotFound:
            pass

    def remove_volume(self, db_volume_name):
        try:
            self._client.volumes.get(db_volume_name).remove(force=True)
            self.logger.info("Removed volume %s" % db_volume_name)
        except docker.errors.NotFound:
            pass

    def create_volume_and_container(self, db_volume_name, db_volume_preperation_container_name) \
            -> Tuple[Volume, Container]:
        db_volume = self._client.volumes.create(db_volume_name)
        volume_preperation_container = \
            self._client.containers.run(
                image="ubuntu:18.04",
                name=db_volume_preperation_container_name,
                auto_remove=True,
                command="sleep infinity",
                detach=True,
                volumes={
                    db_volume.name: {"bind": "/exa", "mode": "rw"}})
        return db_volume, volume_preperation_container

    def upload_init_db_files(self, volume_preperation_container: Container, db_private_network: str):
        file_like_object = io.BytesIO()
        with tarfile.open(fileobj=file_like_object, mode="x") as tar:
            tar.add("build_utils/lib/test_runner/init_db.sh", "init_db.sh")
            self.add_exa_conf(tar, db_private_network)
        volume_preperation_container.put_archive("/", file_like_object.getbuffer().tobytes())

    def add_exa_conf(self, tar: tarfile.TarFile, db_private_network: str):
        with open("ext/EXAConf") as file:
            template_str = file.read()
        template = Template(template_str)

        rendered_template = template.render(private_network=db_private_network)
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
                "Error during preperation of docker-db volume %s got following ouput %s" % (db_volume.name, output))

    def is_database_ready(self, database_container: Container, timeout_in_seconds: int):
        start_time = time.time()
        timeout_over = lambda current_time: current_time - start_time > timeout_in_seconds
        while not timeout_over(time.time()):
            (exit_code, output) = \
                database_container.exec_run(
                    cmd='bash -c "ls /exa/logs/db/DB1/*ConnectionServer*"')
            if exit_code == 0:
                return True
            time.sleep(5)
        return False

    def save_db_log_files_as_gzip_tar(self, path: pathlib.Path, database_container: Container):
        stream, stat = database_container.get_archive("/exa/logs/db")
        with gzip.open(path, "wb") as file:
            for chunk in stream:
                file.write(chunk)
