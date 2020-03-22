class TaskLoggerWrapper():
    def __init__(self, logger, task_id):
        self.task_id = task_id
        self.logger = logger

    def get_message(self, msg):
        return f"{self.task_id}: {msg}"

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(self.get_message(msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(self.get_message(msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(self.get_message(msg), *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        self.logger.warn(self.get_message(msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(self.get_message(msg), *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        self.logger.exception(self.get_message(msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.logger.critical(self.get_message(msg), *args, **kwargs)

    fatal = critical