import luigi

from exaslct_src.lib.test_environment.docker_db_test_environment_parameter import DockerDBTestEnvironmentParameter
from exaslct_src.lib.test_environment.environment_type import EnvironmentType
from exaslct_src.lib.test_environment.external_test_environment_parameter import ExternalDatabaseCredentialsParameter
from exaslct_src.lib.test_environment.general_spawn_test_environment_parameter import GeneralSpawnTestEnvironmentParameter


class SpawnTestEnvironmentParameter(GeneralSpawnTestEnvironmentParameter,
                                    ExternalDatabaseCredentialsParameter,
                                    DockerDBTestEnvironmentParameter):
    environment_type = luigi.EnumParameter(enum=EnvironmentType)