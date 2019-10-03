import luigi
from luigi import Config


class DockerDBTestEnvironmentParameter(Config):
    docker_db_image_name = luigi.OptionalParameter(None)
    docker_db_image_version = luigi.OptionalParameter(None)
    reuse_database = luigi.BoolParameter(False, significant=False)
    database_port_forward = luigi.OptionalParameter(None, significant=False)
    bucketfs_port_forward = luigi.OptionalParameter(None, significant=False)
