import textwrap
from pathlib import Path

import click
from exasol_integration_test_docker_environment.cli.cli import cli
from exasol_integration_test_docker_environment.cli.common import add_options

from exaslct_src.exaslct.cli.options.flavor_options import single_flavor_options
from exaslct_src.exaslct.lib.tasks.upload.language_definition import LanguageDefinition


@cli.command()
@add_options(single_flavor_options)
@click.option('--bucketfs-name', type=str, required=True)
@click.option('--bucket-name', type=str, required=True)
@click.option('--path-in-bucket', type=str, required=False, default="")
@click.option('--container-name', type=str, required=True)
def generate_language_activation(
        flavor_path: str,
        bucketfs_name: str,
        bucket_name: str,
        path_in_bucket: str,
        container_name: str):
    """
    Generate the language activation statement
    """

    language_definition = \
        LanguageDefinition(release_name=container_name,
                           flavor_path=flavor_path,
                           bucketfs_name=bucketfs_name,
                           bucket_name=bucket_name,
                           path_in_bucket=path_in_bucket)

    command_line_output_str = textwrap.dedent(f"""

            In SQL, you can activate the languages supported by the {Path(flavor_path).name}
            flavor by using the following statements:


            To activate the flavor only for the current session:

            {language_definition.generate_alter_session()}


            To activate the flavor on the system:

            {language_definition.generate_alter_system()}
            """)
    print(command_line_output_str)
