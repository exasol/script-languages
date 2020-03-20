from enum import Enum


class EnvironmentType(Enum):
    docker_db = 1,
    external_db = 2