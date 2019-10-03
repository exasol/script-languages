import luigi
from luigi import Config

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask


class UploadContainerParameter(DependencyLoggerBaseTask):
    database_host = luigi.Parameter()
    bucketfs_port = luigi.IntParameter()
    bucketfs_username = luigi.Parameter(significant=False)
    bucketfs_password = luigi.Parameter(significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    bucketfs_name = luigi.Parameter()
    bucket_name = luigi.Parameter()
    path_in_bucket = luigi.Parameter()
    bucketfs_https = luigi.BoolParameter(False)
    release_name = luigi.OptionalParameter()
