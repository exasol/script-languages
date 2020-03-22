import luigi
from luigi import Config


class GeneralSpawnTestEnvironmentParameter(Config):

    reuse_database_setup = luigi.BoolParameter(False, significant=False)
    reuse_test_container = luigi.BoolParameter(False, significant=False)
    no_test_container_cleanup_after_end = luigi.BoolParameter(False, significant=False)
    max_start_attempts = luigi.IntParameter(2, significant=False)
