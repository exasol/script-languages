import getpass
from typing import Callable

import luigi

from exaslct_src.stoppable_task import StoppableTask


def set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory):
    luigi.configuration.get_config().set('build_config', 'force_build', str(force_build))
    luigi.configuration.get_config().set('build_config', 'force_pull', str(force_pull))
    set_output_directory(output_directory)
    if temporary_base_directory is not None:
        luigi.configuration.get_config().set('build_config', 'temporary_base_directory', temporary_base_directory)
    luigi.configuration.get_config().set('build_config', 'log_build_context_content', str(log_build_context_content))


def set_output_directory(output_directory):
    if output_directory is not None:
        luigi.configuration.get_config().set('build_config', 'output_directory', output_directory)


def set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username):
    if docker_base_url is not None:
        luigi.configuration.get_config().set('docker_config', 'base_url', docker_base_url)
    if docker_repository_name is not None:
        luigi.configuration.get_config().set('docker_config', 'repository_name', docker_repository_name)
    if docker_username is not None:
        if docker_password is not None:
            luigi.configuration.get_config().set('docker_config', 'username', docker_username)
            luigi.configuration.get_config().set('docker_config', 'password', docker_password)
        else:
            password = getpass.getpass("Docker password for username %s:" % docker_username)
            luigi.configuration.get_config().set('docker_config', 'username', docker_username)
            luigi.configuration.get_config().set('docker_config', 'password', password)


def run_tasks(tasks, workers,
              on_success: Callable[[], None] = None,
              on_failure: Callable[[], None] = None):
    if StoppableTask.failed_target.exists():
        StoppableTask.failed_target.remove()
    no_scheduling_errors = luigi.build(tasks, workers=workers, local_scheduler=True, log_level="INFO")
    if StoppableTask.failed_target.exists() or not no_scheduling_errors:
        if on_failure is not None:
            on_failure()
        exit(1)
    else:
        if on_success is not None:
            on_success()
        exit(0)

def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options