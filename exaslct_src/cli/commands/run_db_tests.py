import json
from typing import Tuple

import luigi
from click._unicodefun import click

from exaslct_src import TestContainer
from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_config, run_tasks, add_options
from exaslct_src.cli.options \
    import build_options, flavor_options, system_options, release_options, \
    docker_options_login_not_required


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
@add_options(build_options)
@add_options(docker_options_login_not_required)
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
                reuse_database_setup: bool,
                reuse_uploaded_container: bool,
                force_rebuild: bool,
                force_rebuild_from: Tuple[str, ...],
                force_pull: bool,
                output_directory: str,
                temporary_base_directory: str,
                log_build_context_content: bool,
                cache_directory: str,
                docker_repository_name: str,
                docker_username: str,
                docker_password: str,
                workers: int,
                task_dependencies_dot_file: str):
    """
    This command runs the integration tests in local docker-db.
    The systems spawns a test environment in which the test are executed.
    After finishing the tests, the test environment gets cleaned up.
    If the stages or the packaged container do not exists locally,
    the system will build, pull or export them before running the tests.
    """
    set_build_config(force_rebuild,
                     force_rebuild_from,
                     force_pull,
                     log_build_context_content,
                     output_directory,
                     temporary_base_directory,
                     cache_directory)
    set_docker_config(docker_password, docker_repository_name, docker_username)

    tasks = lambda: [TestContainer(flavor_paths=list(flavor_path),
                                   release_types=list([release_type]),
                                   generic_language_tests=list(generic_language_test),
                                   test_folders=list(test_folder),
                                   test_files=list(test_file),
                                   test_restrictions=list(test),
                                   languages=list(test_language),
                                   test_environment_vars=json.loads(test_environment_vars),
                                   test_log_level=test_log_level,
                                   reuse_database=reuse_database,
                                   reuse_uploaded_container=reuse_uploaded_container,
                                   reuse_database_setup=reuse_database_setup
                                   )]

    def on_success():
        target = luigi.LocalTarget(
            "%s/logs/test-runner/db-test/tests/current" % (output_directory))

        print("Test Results:")
        with target.open("r") as f:
            print(f.read())

    run_tasks(tasks, workers, task_dependencies_dot_file, on_success=on_success)
