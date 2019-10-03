from pathlib import Path

from exaslct_src.lib.base.dependency_logger_base_task import DependencyLoggerBaseTask


# This task is needed because ExportContainerTask and SpawnTestContainer
# requires the releases directory which stores the exported container.
# However, we wanted to avoid that SpawnTestContainer depends on ExportContainerTask,
# because ExportContainerTask has a high runtime and SpawnTestContainer is port of SpawnTestEnvironment
# which has a long runtime, too.
class CreateExportDirectory(DependencyLoggerBaseTask):

    def run_task(self):
        export_directory = Path(self.get_cache_path(), "exports")
        export_directory.mkdir(parents=True, exist_ok=True)
        self.return_object(export_directory)
