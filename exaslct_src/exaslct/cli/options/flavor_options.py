import click


def create_flavor_option(multiple):
    help_message = "Path to the directory with the flavor definition.\n" + \
                   "The last segment of the path is used as the name of the flavor."
    if multiple:
        help_addition = "The option can be repeated with different flavors.\n" + \
                        "The system will run the command for each flavor."
        help_message = help_message + "\n" + help_addition

    return click.option('--flavor-path',
                        required=True,
                        multiple=multiple,
                        type=click.Path(exists=True, file_okay=False, dir_okay=True),
                        help=help_message)


flavor_options = [create_flavor_option(multiple=True)]
single_flavor_options = [create_flavor_option(multiple=False)]
