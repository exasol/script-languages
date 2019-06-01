from enum import Enum

import luigi


class WriteLogFilesToConsole(Enum):
    only_error = 1
    all = 2


class log_config(luigi.Config):
    write_log_files_to_console = luigi.EnumParameter(enum=WriteLogFilesToConsole,
                                                     default=WriteLogFilesToConsole.only_error)
    log_task_is_still_running = luigi.BoolParameter(False)
    seconds_between_still_running_logs = luigi.IntParameter(60)
