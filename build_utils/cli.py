import getpass
from typing import Tuple

import luigi
from click._unicodefun import click

from build_utils import DockerBuild, DockerPush, ExportContainer, TestContainer
from build_utils.release_type import str
from build_utils.stoppable_task import StoppableTask
from build_utils.upload_container import UploadContainer


@click.group()
def cli():
    pass


@cli.command()
@click.option('--flavor-path', required=True, multiple=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--force-build/--no-force-build', default=False)
@click.option('--force-pull/--no-force-pull', default=False)
@click.option('--docker-base-url', default=None, type=str)
@click.option('--docker-repository-name', default=None, type=str)
@click.option('--docker-username', type=str)
@click.option('--docker-password', type=str)
@click.option('--output-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--temporary-base-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--log-build-context-content/--no-log-build-context-content', default=False)
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
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [DockerBuild(flavor_paths=list(flavor_path))]
    run_tasks(tasks, workers)


@cli.command()
@click.option('--flavor-path', required=True, multiple=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--force-build/--no-force-build', default=False)
@click.option('--force-pull/--no-force-pull', default=False)
@click.option('--docker-base-url', default=None, type=str)
@click.option('--docker-repository-name', default=None, type=str)
@click.option('--docker-username', type=str)
@click.option('--docker-password', type=str)
@click.option('--output-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--temporary-base-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--log-build-context-content/--no-log-build-context-content', default=False)
@click.option('--workers', default=5, type=int)
def push(flavor_path: Tuple[str, ...],
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
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [DockerPush(flavor_paths=list(flavor_path))]
    run_tasks(tasks, workers)


@cli.command()
@click.option('--flavor-path', required=True, multiple=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--release-type', multiple=True,
              type=click.Choice(['Release', 'BaseTest', "FlavorTest"]))
@click.option('--force-build/--no-force-build', default=False)
@click.option('--force-pull/--no-force-pull', default=False)
@click.option('--docker-base-url', default=None, type=str)
@click.option('--docker-repository-name', default=None, type=str)
@click.option('--docker-username', type=str)
@click.option('--docker-password', type=str)
@click.option('--output-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--temporary-base-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--log-build-context-content/--no-log-build-context-content', default=False)
@click.option('--workers', default=5, type=int)
def export(flavor_path: Tuple[str, ...],
           release_type: Tuple[str, ...],
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
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [ExportContainer(flavor_paths=list(flavor_path), release_types=list(release_type))]
    run_tasks(tasks, workers)


@cli.command()
@click.option('--flavor-path', required=True, multiple=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--release-type', type=click.Choice(['Release', 'BaseTest', "FlavorTest"]))
@click.option('--database-host', type=str, required=True)
@click.option('--bucketfs-port', type=int, required=True)
@click.option('--bucketfs-username', type=str, required=True)
@click.option('--bucketfs-password', type=str)
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucket-name', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=True)
@click.option('--force-build/--no-force-build', default=False)
@click.option('--force-pull/--no-force-pull', default=False)
@click.option('--docker-base-url', default=None, type=str)
@click.option('--docker-repository-name', default=None, type=str)
@click.option('--docker-username', type=str)
@click.option('--docker-password', type=str)
@click.option('--output-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--temporary-base-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--log-build-context-content/--no-log-build-context-content', default=False)
@click.option('--workers', default=5, type=int)
def upload(flavor_path: Tuple[str, ...],
           release_type: str,
           database_host: str,
           bucketfs_port: int,
           bucketfs_username: str,
           bucketfs_password: str,
           bucketfs_name: str,
           bucket_name: str,
           path_in_bucket: str,
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
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [UploadContainer(flavor_paths=list(flavor_path),
                             release_type=release_type,
                             database_host=database_host,
                             bucketfs_port=bucketfs_port,
                             bucketfs_username=bucketfs_username,
                             bucketfs_password=bucketfs_password,
                             bucketfs_name=bucketfs_name,
                             bucket_name=bucket_name,
                             path_in_bucket=path_in_bucket
                             )]
    run_tasks(tasks, workers)


@cli.command()
@click.option('--flavor-path', required=True, multiple=True,
              type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option('--release-type', multiple=True, default=tuple(["Release"]),
              type=click.Choice(['Release', 'BaseTest', "FlavorTest"]))
@click.option('--generic-language-test', multiple=True, type=str)
@click.option('--test-folder', multiple=True, type=click.Path())
@click.option('--test-file', multiple=True, type=click.Path())
@click.option('--test-language', multiple=True, type=str)
@click.option('--test', multiple=True, type=str)
@click.option('--test-environment-vars', type=str)
@click.option('--test-log-level', default="critical",
              type=click.Choice(['critical', 'error', "warning", "info", "debug"]))
@click.option('--reuse-database/--no-reuse-database', default=False)
@click.option('--reuse-uploaded-container/--no-reuse-uploaded-container', default=False)
@click.option('--force-build/--no-force-build', default=False)
@click.option('--force-pull/--no-force-pull', default=False)
@click.option('--docker-base-url', default=None, type=str)
@click.option('--docker-repository-name', default=None, type=str)
@click.option('--docker-username', type=str)
@click.option('--docker-password', type=str)
@click.option('--output-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--temporary-base-directory', default=None, type=click.Path(file_okay=False, dir_okay=True))
@click.option('--log-build-context-content/--no-log-build-context-content', default=False)
@click.option('--workers', default=5, type=int)
def run_db_test(flavor_path: Tuple[str, ...],
                release_type: Tuple[str, ...],
                generic_language_test: Tuple[str, ...],
                test_folder: Tuple[str, ...],
                test_file: Tuple[str, ...],
                test_language: Tuple[str, ...],
                test: Tuple[str, ...],
                test_environment_vars: str,
                test_log_level: str,
                reuse_database: bool,
                reuse_uploaded_container: bool,
                force_build: bool,
                force_pull: bool,
                docker_base_url: str,
                docker_repository_name: str,
                docker_username: str,
                docker_password: str,
                output_directory: str,
                temporary_base_directory: str,
                log_build_context_content: bool,
                workers: int
                ):
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [TestContainer(flavor_paths=list(flavor_path),
                           release_types=list(release_type),
                           generic_language_tests=list(generic_language_test),
                           test_folders=list(test_folder),
                           test_files=list(test_file),
                           test_restrictions=list(test),
                           languages=list(test_language),
                           test_environment_vars=test_environment_vars,
                           test_log_level=test_log_level,
                           reuse_database=reuse_database,
                           reuse_uploaded_container=reuse_uploaded_container
                           )]
    run_tasks(tasks, workers)


def set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory):
    luigi.configuration.get_config().set('build_config', 'force_build', str(force_build))
    luigi.configuration.get_config().set('build_config', 'force_pull', str(force_pull))
    if output_directory is not None:
        luigi.configuration.get_config().set('build_config', 'output_directory', output_directory)
    if temporary_base_directory is not None:
        luigi.configuration.get_config().set('build_config', 'temporary_base_directory', temporary_base_directory)
    luigi.configuration.get_config().set('build_config', 'log_build_context_content', str(log_build_context_content))


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


def run_tasks(tasks, workers):
    if StoppableTask.failed_target.exists():
        StoppableTask.failed_target.remove()
    luigi.build(tasks, workers=workers, local_scheduler=True, log_level="INFO")
    if StoppableTask.failed_target.exists():
        exit(1)
    else:
        exit(0)


if __name__ == '__main__':
    cli()
