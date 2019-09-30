import luigi
from luigi.parameter import ParameterVisibility

class ExternalDatabaseHostParameter():
    external_exasol_db_host = luigi.OptionalParameter()
    external_exasol_db_port = luigi.OptionalParameter()
    external_exasol_bucketfs_port = luigi.Parameter()

class ExternalTestEnvironmentParameter(ExternalDatabaseHostParameter):
    external_exasol_db_user = luigi.OptionalParameter()
    external_exasol_db_password = luigi.OptionalParameter(significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)
    external_exasol_bucketfs_write_password = luigi.OptionalParameter(significant=False, visibility=luigi.parameter.ParameterVisibility.HIDDEN)