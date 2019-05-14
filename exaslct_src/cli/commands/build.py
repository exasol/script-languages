from pathlib import Path
from typing import Tuple

from exaslct_src.cli.cli import cli
from exaslct_src.cli.common import set_build_config, set_docker_config, run_tasks, add_options, import_build_steps
from exaslct_src.cli.options \
    import build_options, flavor_options, system_options, docker_options_login_not_required, goal_options
from exaslct_src.lib.docker_build import DockerBuild


@cli.command()
@add_options(flavor_options)
@add_options(goal_options)
@add_options(build_options)
@add_options(docker_options_login_not_required)
@add_options(system_options)
def build(flavor_path: Tuple[str, ...],
          goal: Tuple[str, ...],
          force_rebuild: bool,
          force_rebuild_from: Tuple[str, ...],
          force_pull: bool,
          output_directory: str,
          temporary_base_directory: str,
          log_build_context_content: bool,
          cache_directory:str,
          docker_repository_name: str,
          docker_username: str,
          docker_password: str,
          workers: int,
          task_dependencies_dot_file: str):
    """
    This command builds all stages of the script language container flavor.
    If stages are cached in a docker registry, they command is going to pull them,
    instead of building them.
    """
    import_build_steps(flavor_path)
    set_build_config(force_rebuild,
                     force_rebuild_from,
                     force_pull,
                     log_build_context_content,
                     output_directory,
                     temporary_base_directory,
                     cache_directory)
    set_docker_config(docker_password, docker_repository_name, docker_username)
    tasks = lambda: [DockerBuild(flavor_paths=list(flavor_path), goals=list(goal))]
    run_tasks(tasks, workers, task_dependencies_dot_file)

