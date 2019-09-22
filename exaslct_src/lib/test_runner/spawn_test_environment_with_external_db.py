import luigi

from exaslct_src.lib.test_runner.spawn_test_environment import SpawnTestEnvironment


class SpawnTestEnvironmentWithExternalDB(SpawnTestEnvironment):

    external_exasol_db_host = luigi.OptionalParameter(None)
    external_exasol_db_port = luigi.OptionalParameter(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def create_network_task(self, attempt):
        return

    def create_spawn_database_task(self, network_info_dict, attempt):
        return