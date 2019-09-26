import luigi

from exaslct_src.lib.test_runner.docker_db_test_environment_parameter import DockerDBTestEnvironmentParameter
from exaslct_src.lib.test_runner.environment_type import EnvironmentType
from exaslct_src.lib.test_runner.external_test_environment_parameter import ExternalTestEnvironmentParameter
from exaslct_src.lib.test_runner.general_spawn_test_environment_parameter import GeneralSpawnTestEnvironmentParameter


class SpawnTestEnvironmentParameter(GeneralSpawnTestEnvironmentParameter,
                                    ExternalTestEnvironmentParameter,
                                    DockerDBTestEnvironmentParameter):
    environment_type = luigi.EnumParameter(enum=EnvironmentType)