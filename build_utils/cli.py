import getpass
import json
from typing import Tuple

import luigi
from click._unicodefun import click

from build_utils import DockerBuild, DockerPush, ExportContainer, TestContainer, UploadContainer
from build_utils.stoppable_task import StoppableTask

flavor_options = [
    click.option('--flavor-path',
                 required=True,
                 multiple=True,
                 type=click.Path(exists=True, file_okay=False, dir_okay=True),
                 help="Path to the directory with the flavor definition. "
                      "The last segment of the path is used as the name of the flavor. "
                      "The option can be repeated with different flavors. "
                      "The system will run the command for each flavor.")
]

docker_options = [
    click.option('--docker-base-url', type=str,
                 default="unix:///var/run/docker.sock",
                 show_default=True,
                 help="URL to the socket of the docker daemon."),
    click.option('--docker-repository-name', type=str,
                 default="exasol/script-language-container",
                 show_default=True,
                 help="Name of the docker repository for naming, pushing or fetching cached stages. "
                      "The repository name may contain URL of the docker registory, "
                      "the username and the actual repository name. "
                      "A common strcuture is <docker-registry-url>/<username>/<repository-name>"),
    click.option('--docker-username', type=str,
                 help="Username for the docker registry from where the system pulls cached stages."),
    click.option('--docker-password', type=str,
                 help="Password for the docker registry from where the system pulls cached stages. "
                      "Without password option the system prompts for the password."),
]

build_options = [
    click.option('--force-build/--no-force-build', default=False,
                 help="Forces the system to complete rebuild of a all stages."),
    click.option('--force-pull/--no-force-pull', default=False,
                 help="Forces the system to pull all stages if available, otherwise it rebuilds a stage."),

    click.option('--output-directory',
                 type=click.Path(file_okay=False, dir_okay=True),
                 default=".build_ouptuts",
                 show_default=True,
                 help="Output directory where the system stores all output and log files."),
    click.option('--temporary-base-directory',
                 type=click.Path(file_okay=False, dir_okay=True),
                 default="/tmp",
                 show_default=True,
                 help="Directory where the system creates temporary directories."
                 ),
    click.option('--log-build-context-content/--no-log-build-context-content',
                 default=False,
                 help="For Debugging: Logs the files and directories in the build context of a stage"),
]

system_options = [
    click.option('--workers', type=int,
                 default=5, show_default=True,
                 help="Number of parallel workers")
]

release_options = [
    click.option('--release-type',
                 type=click.Choice(['Release', 'BaseTest', "FlavorTest"]),
                 )
]


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@click.group()
def cli():
    pass


@cli.command()
@add_options(flavor_options)
@add_options(build_options)
@add_options(docker_options)
@add_options(system_options)
def build(flavor_path: Tuple[str, ...],
          force_build: bool,
          force_pull: bool,
          output_directory: str,
          temporary_base_directory: str,
          log_build_context_content: bool,
          docker_base_url: str,
          docker_repository_name: str,
          docker_username: str,
          docker_password: str,
          workers: int):
    """
    This command builds all stages of the script language container flavor.
    If stages are cached in a docker registry, they command is going to pull them,
    instead of building them.
    """
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [DockerBuild(flavor_paths=list(flavor_path))]
    run_tasks(tasks, workers)


@cli.command()
@add_options(flavor_options)
@add_options(build_options)
@add_options(docker_options)
@add_options(system_options)
def push(flavor_path: Tuple[str, ...],
         force_build: bool,
         force_pull: bool,
         output_directory: str,
         temporary_base_directory: str,
         log_build_context_content: bool,
         docker_base_url: str,
         docker_repository_name: str,
         docker_username: str,
         docker_password: str,
         workers: int):
    """
    This command pushes all stages of the script language container flavor.
    If the stages do not exists locally, the system will build or pull them before the push.
    """
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [DockerPush(flavor_paths=list(flavor_path))]
    run_tasks(tasks, workers)


@cli.command()
@add_options(flavor_options)
@add_options(build_options)
@add_options(docker_options)
@add_options(system_options)
def export(flavor_path: Tuple[str, ...],
           release_type: str,
           force_build: bool,
           force_pull: bool,
           output_directory: str,
           temporary_base_directory: str,
           log_build_context_content: bool,
           docker_base_url: str,
           docker_repository_name: str,
           docker_username: str,
           docker_password: str,
           workers: int):
    """
    This command exports the whole script language container package of the flavor,
    ready for the upload into the bucketfs. If the stages do not exists locally,
    the system will build or pull them before the exporting the packaged container.
    """
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)
    tasks = [ExportContainer(flavor_paths=list(flavor_path), release_types=list(release_type))]
    run_tasks(tasks, workers)


@cli.command()
@add_options(flavor_options)
@add_options(release_options)
@click.option('--database-host', type=str,
              required=True)
@click.option('--bucketfs-port', type=int, required=True)
@click.option('--bucketfs-username', type=str, required=True)
@click.option('--bucketfs-password', type=str)
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucket-name', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=True)
@add_options(build_options)
@add_options(docker_options)
@add_options(system_options)
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
           output_directory: str,
           temporary_base_directory: str,
           log_build_context_content: bool,
           docker_base_url: str,
           docker_repository_name: str,
           docker_username: str,
           docker_password: str,
           workers: int):
    """
    This command uploads the whole script language container package of the flavor to the database.
    If the stages or the packaged container do not exists locally, the system will build, pull or
    export them before the upload.
    """
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
@add_options(flavor_options)
@add_options(release_options)
@click.option('--generic-language-test', multiple=True, type=str,
              help="Specifies for which languages the test runner executes generic language tests."
                   "The option can be repeated with different languages. "
                   "The test runner will run the generic language test for each language."
              )
@click.option('--test-folder', multiple=True, type=click.Path(),
              help="Specifies in which directories the test runners looks for test files to execute."
                   "The option can be repeated with different directories. "
                   "The test runner will run the test files in each of these directories."
              )
@click.option('--test-file', multiple=True, type=click.Path(),
              help="Specifies in which test-files the test runners should execute."
                   "The option can be repeated with different test files. "
                   "The test runner will run all specified test files."
              )
@click.option('--test-language', multiple=True, type=str,
              help="Specifies with which language the test files get executed."
                   "The option can be repeated with different languages. "
                   "The test runner will run the test files with all specified languages."
              )
@click.option('--test', multiple=True, type=str,
              help="Define restriction which tests in the test files should be executed."
                   "The option can be repeated with different restrictions. "
                   "The test runner will run the test files with all specified restrictions."
              )
@click.option('--test-environment-vars', type=str, default="""{"TRAVIS": ""}""",
              show_default=True,
              help="""Specifies the environment variables for the test runner as a json 
              in the form of {"<variable_name>":<value>}.""")
@click.option('--test-log-level', default="critical",
              type=click.Choice(['critical', 'error', "warning", "info", "debug"]),
              show_default=True)
@click.option('--reuse-database/--no-reuse-database', default=False,
              help="Reuse a previous create test-database and "
                   "disables the clean up of the test-database to allow reuse later.")
@click.option('--reuse-uploaded-container/--no-reuse-uploaded-container', default=False,
              help="Reuse the uploaded script-langauge-container in a reused database.")
@add_options(build_options)
@add_options(docker_options)
@add_options(system_options)
def run_db_test(flavor_path: Tuple[str, ...],
                release_type: str,
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
                output_directory: str,
                temporary_base_directory: str,
                log_build_context_content: bool,
                docker_base_url: str,
                docker_repository_name: str,
                docker_username: str,
                docker_password: str,
                workers: int
                ):
    """
    This command runs the integration tests in local docker-db.
    The systems spawns a test environment in which the test are executed.
    After finishing the tests, the test environment gets cleaned up.
    If the stages or the packaged container do not exists locally,
    the system will build, pull or export them before running the tests.
    """
    set_build_config(force_build, force_pull, log_build_context_content, output_directory, temporary_base_directory)
    set_docker_config(docker_base_url, docker_password, docker_repository_name, docker_username)

    tasks = [TestContainer(flavor_paths=list(flavor_path),
                           release_types=list([release_type]),
                           generic_language_tests=list(generic_language_test),
                           test_folders=list(test_folder),
                           test_files=list(test_file),
                           test_restrictions=list(test),
                           languages=list(test_language),
                           test_environment_vars=json.loads(test_environment_vars),
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
