import io
import tarfile
import time

import docker
import luigi

from build_utils.lib.build_config import build_config
from build_utils.lib.docker_config import docker_config


class SpawnTestDatabase(luigi.Task):
    environment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()
        self._docker_config = docker_config()
        self._client = docker.DockerClient(base_url=self._docker_config.base_url)
        self._prepare_outputs()

    def __del__(self):
        self._client.close()

    def _prepare_outputs(self):
        self._database_info_target = luigi.LocalTarget(
            "%s/test-runner/db-test/test-environment/%s/database_info"
            % (self._build_config.ouput_directory,
               self.environment_name))
        if self._database_info_target.exists():
            self._database_info_target.remove()

    def output(self):
        return self._database_info_target

    def run(self):
        database_container_name = f"""db_container_{str(self.task_id)}"""
        try:
            self._client.containers.get(database_container_name).remove(force=True)
            print("remove",database_container_name)
        except docker.errors.NotFound:
            pass
        db_volume = self.prepare_docker_db_volume()
        database_container = self._client.containers.run(image="exasol/docker-db:6.0.12-d1",
                                                name=database_container_name,
                                                auto_remove=True,
                                                detach=True,
                                                privileged=True,
                                                volumes={db_volume.name: {"bind": "/exa", "mode": "rw"}})

        finished = False
        while not finished:
            (exit_code, output) = database_container.exec_run(cmd='bash -c "ls /exa/logs/db/DB1/*ConnectionServer*"')
            print(output)
            if exit_code==0:
                finished=True
            time.sleep(5)

        # for line in container.logs(follow=True,stream=True):
        #     print(line)
        with self.output().open("w") as file:
            file.write("test")

    def prepare_docker_db_volume(self):
        db_volume_preperation_container_name = f"""db_volumne_preparation_container_{str(self.task_id)}"""
        db_volume_name = f"""db_volumne_{str(self.task_id)}"""
        try:
            self._client.volumes.get(db_volume_name).remove(force=True)
            print("remove", db_volume_name)
        except docker.errors.NotFound:
            pass
        try:
            self._client.containers.get(db_volume_preperation_container_name).remove(force=True)
            print("remove", db_volume_preperation_container_name)
        except docker.errors.NotFound:
            pass
        db_volume, volume_preperation_container = \
            volume_preperation_container, volume_preperation_container = \
            self.create_volumne_and_container(db_volume_name,
                                              db_volume_preperation_container_name)
        try:
            self.upload_init_db_files(volume_preperation_container)
            self.execute_init_db(db_volume, volume_preperation_container)
            return db_volume
        finally:
            volume_preperation_container.remove(force=True)

    def create_volumne_and_container(self, db_volume_name, db_volume_preperation_container_name):
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

    def execute_init_db(self, db_volume, volume_preperation_container):
        (exit_code, output) = volume_preperation_container.exec_run(cmd="bash /init_db.sh")
        if exit_code != 0:
            raise Exception(
                "Error during preperation of docker-db volume %s got following ouput %s" % (db_volume.name, output))

    def upload_init_db_files(self, volume_preperation_container):
        file_like_object = io.BytesIO()
        with tarfile.open(fileobj=file_like_object, mode="x") as tar:
            tar.add("build_utils/lib/test_runner/init_db.sh", "init_db.sh")
            tar.add("ext/EXAConf", "EXAConf")
        volume_preperation_container.put_archive("/", file_like_object.getbuffer().tobytes())
