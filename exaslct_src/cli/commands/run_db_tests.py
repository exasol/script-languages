import json
from typing import Tuple

import luigi
from click._unicodefun import click

from exaslct_src import TestContainer
from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_repository_config, run_task, add_options, \
    import_build_steps, set_job_id
from exaslct_src.cli.options \
    import build_options, flavor_options, system_options, release_options, \
    docker_repository_options, docker_db_options, test_environment_options, external_db_options
from exaslct_src.lib.test_runner.environment_type import EnvironmentType


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
@click.option('--test-language', multiple=True, type=str, default=[None],
              help="Specifies with which language the test files get executed."
                   "The option can be repeated with different languages. "
                   "The test runner will run the test files with all specified languages."
              )
@click.option('--test', multiple=True, type=str,
              help="Define restriction which tests in the test files should be executed."
                   "The option can be repeated with different restrictions. "
                   "The test runner will run the test files with all specified restrictions."
              )
@add_options(test_environment_options)
@add_options(docker_db_options)
@add_options(external_db_options)
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
@click.option('--reuse-database-setup/--no-reuse-database-setup', default=False,
              help="Reuse a previous executed database setup in a reused database")
@click.option('--reuse-uploaded-container/--no-reuse-uploaded-container', default=False,
              help="Reuse the uploaded script-langauge-container in a reused database.")
@click.option('--reuse-test-container/--no-reuse-test-container', default=False,
              help="Reuse the test container which is used for test execution.")
@click.option('--reuse-test-environment/--no-reuse-test-environment', default=False,
              help="Reuse the whole test environment with docker network, test container, "
                   "database, database setup and uploaded container")
@add_options(build_options)
@add_options(docker_repository_options)
@add_options(system_options)
def run_db_test(flavor_path: Tuple[str, ...],
                release_goal: str,
                generic_language_test: Tuple[str, ...],
                test_folder: Tuple[str, ...],
                test_file: Tuple[str, ...],
                test_language: Tuple[str, ...],
                test: Tuple[str, ...],
                environment_type: str,
                max_start_attempts: int,
                docker_db_image_version: str,
                docker_db_image_name: str,
                external_exasol_db_host: str,
                external_exasol_db_port: int,
                external_exasol_bucketfs_port: int,
                external_exasol_db_user: str,
                external_exasol_db_password: str,
                external_exasol_bucketfs_write_password: str,
                test_environment_vars: str,
                test_log_level: str,
                reuse_database: bool,
                reuse_database_setup: bool,
                reuse_uploaded_container: bool,
                reuse_test_container: bool,
                reuse_test_environment: bool,
                force_rebuild: bool,
                force_rebuild_from: Tuple[str, ...],
                force_pull: bool,
                output_directory: str,
                temporary_base_directory: str,
                log_build_context_content: bool,
                cache_directory: str,
                build_name: str,
                source_docker_repository_name: str,
                source_docker_tag_prefix: str,
                source_docker_username: str,
                source_docker_password: str,
                target_docker_repository_name: str,
                target_docker_tag_prefix: str,
                target_docker_username: str,
                target_docker_password: str,
                workers: int,
                task_dependencies_dot_file: str):
    """
    This command runs the integration tests in local docker-db.
    The systems spawns a test environment in which the test are executed.
    After finishing the tests, the test environment gets cleaned up.
    If the stages or the packaged container do not exists locally,
    the system will build, pull or export them before running the tests.
    """
    import_build_steps(flavor_path)
    set_build_config(force_rebuild,
                     force_rebuild_from,
                     force_pull,
                     log_build_context_content,
                     output_directory,
                     temporary_base_directory,
                     cache_directory,
                     build_name)
    set_docker_repository_config(source_docker_password, source_docker_repository_name, source_docker_username,
                                 source_docker_tag_prefix, "source")
    set_docker_repository_config(target_docker_password, target_docker_repository_name, target_docker_username,
                                 target_docker_tag_prefix, "target")

    if reuse_test_environment:
        reuse_database = True
        reuse_uploaded_container = True
        reuse_test_container = True
        reuse_database_setup = True
    if environment_type == EnvironmentType.external_db.name:
        if external_exasol_db_host is None:
            handle_commandline_error("Commandline parameter --external-exasol-db-host not set")
        if external_exasol_db_port is None:
            handle_commandline_error("Commandline parameter --external-exasol_db-port not set")
        if external_exasol_bucketfs_port is None:
            handle_commandline_error("Commandline parameter --external-exasol-bucketfs-port not set")
    set_job_id(TestContainer.__name__)
    task_creator = lambda: TestContainer(flavor_paths=list(flavor_path),
                                         release_goals=list(release_goal),
                                         generic_language_tests=list(generic_language_test),
                                         test_folders=list(test_folder),
                                         test_files=list(test_file),
                                         test_restrictions=list(test),
                                         languages=list(test_language),
                                         test_environment_vars=json.loads(test_environment_vars),
                                         test_log_level=test_log_level,
                                         reuse_uploaded_container=reuse_uploaded_container,
                                         environment_type=EnvironmentType[environment_type],
                                         reuse_database_setup=reuse_database_setup,
                                         reuse_test_container=reuse_test_container,
                                         docker_db_image_name=docker_db_image_name,
                                         docker_db_image_version=docker_db_image_version,
                                         reuse_database=reuse_database,
                                         max_start_attempts=max_start_attempts,
                                         external_exasol_db_host=external_exasol_db_host,
                                         external_exasol_db_port=external_exasol_db_port,
                                         external_exasol_bucketfs_port=external_exasol_bucketfs_port,
                                         external_exasol_db_user=external_exasol_db_user,
                                         external_exasol_db_password=external_exasol_db_password,
                                         external_exasol_bucketfs_write_password=external_exasol_bucketfs_write_password)
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if success:
        print("Test Results:")
        with task.command_line_output_target.open("r") as f:
            print(f.read())
    else:
        exit(1)


def handle_commandline_error(error):
    print(error)
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    exit(1)
