import click

from exaslct_src.cli.options.system_options import output_directory_option, tempory_base_directory_option

build_options = [
    click.option('--force-rebuild/--no-force-rebuild', default=False,
                 help="Forces the system to complete rebuild all stages down to the stages "
                      "specified with the options --force-rebuild-from."),
    click.option('--force-rebuild-from', multiple=True, type=str,
                 help="If the option --force-rebuild is given, "
                      "this options specifies for which stages and dependent stages system will force a rebuild. "
                      "The option can be repeated with different stages. "
                      "The system will than force the rebuild of these stages and their. dependet stages."
                 ),
    click.option('--force-pull/--no-force-pull', default=False,
                 help="Forces the system to pull all stages if available, otherwise it rebuilds a stage."),
    output_directory_option,
    tempory_base_directory_option,
    click.option('--log-build-context-content/--no-log-build-context-content',
                 default=False,
                 help="For Debugging: Logs the files and directories in the build context of a stage"),
    click.option('--cache-directory', default=None, type=click.Path(file_okay=False, dir_okay=True, exists=False),
                 help="Directory from where saved docker images can be loaded"),
    click.option('--build-name', default=None, type=str,
                 help="Name of the build. For example: Repository + CI Build Number"),
]
