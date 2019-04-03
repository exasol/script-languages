import luigi

class log_config(luigi.Config):

    write_log_files_to_console = luigi.BoolParameter(True)
    log_task_is_still_running = luigi.BoolParameter(True)
    seconds_between_still_running_logs = luigi.IntParameter(60)
