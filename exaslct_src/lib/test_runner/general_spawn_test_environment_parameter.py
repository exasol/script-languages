import luigi
from luigi import Config
from luigi.parameter import ParameterVisibility

from exaslct_src.lib.test_runner.environment_type import EnvironmentType


class GeneralSpawnTestEnvironmentParameter(Config):

    reuse_database_setup = luigi.BoolParameter(False, significant=False)
    reuse_test_container = luigi.BoolParameter(False, significant=False)

    max_start_attempts = luigi.IntParameter(2, significant=False)