from exaslct_src.lib.base.task_state import TaskState


class WrongTaskStateException(Exception):

    def __init__(self, task_state: TaskState, method: str):
        super().__init__(f"Calling method {method} in task state {task_state} not allowed")