import click

goal_options = [
    click.option('--goal', multiple=True, type=str,
                 help="Selects which build stage will be build or pushed. "
                      "The system will build also all dependencies of the selected build stage. "
                      "The option can be repeated with different stages. "
                      "The system will than build all these stages and their dependencies."
                 )]

release_options = [
    click.option('--release-goal',
                 type=str,
                 default=["release"],
                 multiple=True
                 )
]
