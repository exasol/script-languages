import logging
import pathlib
from datetime import datetime
from typing import Generator, List

import luigi
from luigi import LocalTarget

from exaslct_src.lib.build_config import build_config
from exaslct_src.lib.still_running_logger import StillRunningLogger, StillRunningLoggerThread
from exaslct_src.lib.task_dependency import TaskDescription, TaskDependency, DependencyType, DependencyState


class StoppingFurtherExecution(Exception):
    pass
