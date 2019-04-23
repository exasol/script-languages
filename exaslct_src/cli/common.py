import getpass
import shutil
from datetime import datetime
from typing import Callable, List

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
            password = getpass.getpass("Docker Registry Password for User %s:" % docker_username)
            luigi.configuration.get_config().set('docker_config', 'username', docker_username)
            luigi.configuration.get_config().set('docker_config', 'password', password)


def run_tasks(tasks_creator: Callable[[], List[luigi.Task]],
              workers: int,
              on_success: Callable[[], None] = None,
              on_failure: Callable[[], None] = None):
    setup_worker()
    start_time = datetime.now()
    tasks = remove_stoppable_task_targets(tasks_creator)
    no_scheduling_errors = luigi.build(tasks, workers=workers, local_scheduler=True, log_level="INFO")
    if StoppableTask.failed_target.exists() or not no_scheduling_errors:
        handle_failure(on_failure)
    else:
        handle_success(on_success, start_time)


def handle_success(on_success, start_time):
    if on_success is not None:
        on_success()
    timedelta = datetime.now() - start_time
    print("The command took %s s" % timedelta.total_seconds())
    exit(0)


def handle_failure(on_failure):
    if on_failure is not None:
        on_failure()
    exit(1)


def remove_stoppable_task_targets(tasks_creator):
    if StoppableTask.failed_target.exists():
        StoppableTask.failed_target.remove()
    if StoppableTask.timers_dir.exists():
        shutil.rmtree(str(StoppableTask.timers_dir))
    tasks = tasks_creator()
    return tasks


def setup_worker():
    luigi.configuration.get_config().set('worker', 'wait_interval', str(0.1))
    luigi.configuration.get_config().set('worker', 'wait_jitter', str(0.5))


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options
