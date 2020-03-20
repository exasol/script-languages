from typing import Tuple

from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_repository_config, run_task, add_options, \
    set_job_id
from exaslct_src.cli.options.build_options import build_options
from exaslct_src.cli.options.docker_repository_options import docker_repository_options
from exaslct_src.cli.options.system_options import system_options

from exaslct_src.lib.test_environment.analyze_test_container import DockerTestContainerBuild, AnalyzeTestContainer


@cli.command()
@add_options(build_options)
@add_options(docker_repository_options)
@add_options(system_options)
def build_test_container(
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
    This command builds all stages of the test container for the test environment.
    If stages are cached in a docker registry, they command is going to pull them,
    instead of building them.
    """
    set_build_config(force_rebuild,
                     force_rebuild_from,
                     force_pull,
                     log_build_context_content,
                     output_directory,
                     temporary_base_directory,
                     cache_directory,
                     build_name)
    # Use AnalyzeTestContainer to ensure that all luigi processes got it loaded
    analyze_task = AnalyzeTestContainer.__class__.__name__

    set_docker_repository_config(source_docker_password, source_docker_repository_name, source_docker_username,
                                 source_docker_tag_prefix, "source")
    set_docker_repository_config(target_docker_password, target_docker_repository_name, target_docker_username,
                                 target_docker_tag_prefix, "target")
    set_job_id(DockerTestContainerBuild.__name__)
    task_creator = lambda: DockerTestContainerBuild()
    success, task = run_task(task_creator, workers, task_dependencies_dot_file)
    if not success:
        exit(1)
