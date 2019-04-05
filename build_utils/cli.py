import getpass
import multiprocessing
from typing import Tuple

import luigi
from click._unicodefun import click
from luigi import LocalTarget

from build_utils import DockerBuild
from build_utils.stoppable_task import StoppableTask

@click.group()
def cli():
    pass


@cli.command()
@click.option('--flavor-path', required=True, multiple=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--force-build', is_flag=True)
@click.option('--force-pull', is_flag=True)
@click.option('--docker-base-url', default=None, type=str)
@click.option('--docker-repository-name', default=None, type=str)
@click.option('--docker-username', type=str)
@click.option('--docker-password', type=str)
@click.option('--output-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--temporary-base-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--log-build-context-content', is_flag=True)
@click.option('--workers', default=5, type=int)
def build(flavor_path: Tuple[str, ...],
          force_build: bool,
          force_pull: bool,
          docker_base_url: str,
          docker_repository_name: str,
          docker_username: str,
          docker_password: str,
          output_directory: str,
          temporary_base_directory: str,
          log_build_context_content: bool,
          workers: int):
    if force_build:
        luigi.configuration.get_config().set('build_config', 'force_build', str(force_build))
    if force_pull:
        luigi.configuration.get_config().set('build_config', 'force_pull', str(force_pull))
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
    if output_directory is not None:
        luigi.configuration.get_config().set('build_config', 'output_directory', output_directory)
    if temporary_base_directory is not None:
        luigi.configuration.get_config().set('build_config', 'temporary_base_directory', temporary_base_directory)
    if log_build_context_content:
        luigi.configuration.get_config().set('build_config', 'log_build_context_content',
                                             str(log_build_context_content))
    if StoppableTask.failed_target.exists():
        StoppableTask.failed_target.remove()
    luigi.build([DockerBuild(flavor_paths=list(flavor_path))], workers=workers, local_scheduler=True, log_level="INFO")
    if StoppableTask.failed_target.exists():
        exit(1)
    else:
        exit(0)

if __name__ == '__main__':
    cli()
