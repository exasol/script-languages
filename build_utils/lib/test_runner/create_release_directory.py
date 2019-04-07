import luigi

from build_utils.lib.build_config import build_config
from build_utils.stoppable_task import StoppableTask

# This task is needed because ExportContainerTask and SpawnTestContainer
# requires the releases directory which stores the exported container.
# However, we wanted to avoid that SpawnTestContainer depends on ExportContainerTask,
# because ExportContainerTask has a high runtime and SpawnTestContainer is port of SpawnTestEnvironment
# which has a long runtime, too.
class CreateReleaseDirectory(StoppableTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._build_config = build_config()

    def output(self):
        self.directory = "%s/releases/" % self._build_config.output_directory
        release_directory = luigi.LocalTarget(self.directory + ".created")
        return release_directory

    def run(self):
        with self.output().open("w") as f:
            f.write(self.directory)
