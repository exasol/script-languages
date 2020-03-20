import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Generator, Any, Union

import luigi
import six
from luigi import Task, util
from luigi.parameter import ParameterVisibility
from luigi.task import TASK_ID_TRUNCATE_HASH

from exaslct_src.abstract_method_exception import AbstractMethodException
from exaslct_src.exaslct.lib.config.build_config import build_config
from exaslct_src.test_environment.base.abstract_task_future import AbstractTaskFuture, DEFAULT_RETURN_OBJECT_NAME
from exaslct_src.test_environment.base.job_config import job_config
from exaslct_src.test_environment.base.pickle_target import PickleTarget
from exaslct_src.test_environment.base.task_logger_wrapper import TaskLoggerWrapper
from exaslct_src.test_environment.base.task_state import TaskState
from exaslct_src.test_environment.base.wrong_task_state_exception import WrongTaskStateException

RETURN_TARGETS = "return_targets"

COMPLETION_TARGET = "completion_target"

RUN_DEPENDENCIES = "run_dependencies"


class RequiresTaskFuture(AbstractTaskFuture):

    def __init__(self, task: "BaseTask", index: int):
        self._index = index
        self._task = task
        self._outputs_dict = None

    def get_output(self, name: str = DEFAULT_RETURN_OBJECT_NAME):
        return self._get_outputs_dict()[name].read()

    def list_outputs(self) -> List[str]:
        return list(self._get_outputs_dict().keys())

    def _get_outputs_dict(self) -> Dict[str, PickleTarget]:
        if self._task._task_state == TaskState.RUN:
            if self._outputs_dict is None:
                completion_target = self._task.input()[self._index]
                self._outputs_dict = completion_target.read()
            return self._outputs_dict
        else:
            raise WrongTaskStateException(self._task._task_state, "RequiresTaskFuture.read_outputs_dict")


class RunTaskFuture(AbstractTaskFuture):

    def __init__(self, completion_target: PickleTarget):
        self._outputs_dict = None
        self.completion_target = completion_target

    def get_output(self, name: str = DEFAULT_RETURN_OBJECT_NAME):
        return self._get_outputs_dict()[name].read()

    def list_outputs(self) -> List[str]:
        return list(self._get_outputs_dict().keys())

    def _get_outputs_dict(self) -> Dict[str, PickleTarget]:
        if self._outputs_dict is None:
            self._outputs_dict = self.completion_target.read()
        return self._outputs_dict


class BaseTask(Task):
    caller_output_path = luigi.ListParameter([], significant=False, visibility=ParameterVisibility.HIDDEN)

    def __init__(self, *args, **kwargs):
        self._registered_tasks = []
        self._run_dependencies_tasks = []
        self._registered_return_targets = {}
        self._task_state = TaskState.INIT
        super().__init__(*args, **kwargs)
        self.task_id = self.task_id_str(self.get_task_family(),
                                        self.get_parameter_as_string_dict())
        self.__hash = hash(self.task_id)
        self._init_non_pickle_attributes()
        self.register_required()
        self._task_state = TaskState.NONE

    def _init_non_pickle_attributes(self):
        logger = logging.getLogger(f'luigi-interface.{self.__class__.__name__}')
        self.logger = TaskLoggerWrapper(logger, self.__repr__())
        self._complete_target = PickleTarget(path=self._get_tmp_path_for_completion_target())
        self._run_dependencies_target = PickleTarget(path=self._get_tmp_path_for_run_dependencies())

    def __getstate__(self):
        new_dict = dict(self.__dict__)
        del new_dict["logger"]
        del new_dict["_complete_target"]
        del new_dict["_run_dependencies_target"]
        return new_dict

    def __setstate__(self, new_dict):
        self.__dict__ = new_dict
        self._init_non_pickle_attributes()

    def task_id_str(self, task_family, params):
        """
        Returns a canonical string used to identify a particular task

        :param task_family: The task family (class name) of the task
        :param params: a dict mapping parameter names to their serialized values
        :return: A unique, shortened identifier corresponding to the family and params
        """
        # task_id is a concatenation of task family, the first values of the first 3 parameters
        # sorted by parameter name and a md5hash of the family/parameters as a cananocalised json.
        param_str = json.dumps(params, separators=(',', ':'), sort_keys=True)
        hash_input = job_config().job_id + param_str
        param_hash = hashlib.sha3_256(hash_input.encode('utf-8')).hexdigest()
        return '{}_{}'.format(task_family, param_hash[:TASK_ID_TRUNCATE_HASH])

    def get_parameter_as_string_dict(self):
        """
        Convert all parameters to a str->str hash.
        """
        params_str = {}
        params = dict(self.get_params())
        for param_name, param_value in six.iteritems(self.param_kwargs):
            if (params[param_name].significant):
                params_str[param_name] = params[param_name].serialize(param_value)
        return params_str

    def get_output_path(self) -> Path:
        path = Path(self._get_output_path_for_job(),
                    "outputs",
                    Path(*self._extend_output_path()))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _get_output_path_for_job(self) -> Path:
        return Path(build_config().output_directory,
                    "jobs", job_config().job_id)

    def _extend_output_path(self):
        extension = self.extend_output_path()
        if extension is None or extension == []:
            return self.task_id
        else:
            return extension

    def extend_output_path(self):
        return list(self.caller_output_path) + [self.task_id]

    def _get_tmp_path_for_job(self) -> Path:
        return Path(self._get_output_path_for_job(), "temp")

    def _get_tmp_path_for_task(self) -> Path:
        return Path(self._get_tmp_path_for_job(),
                    self.task_id)

    def _get_tmp_path_for_returns(self, name: str) -> Path:
        return Path(self._get_tmp_path_for_task(), RETURN_TARGETS, name)

    def _get_tmp_path_for_completion_target(self) -> Path:
        return Path(self._get_tmp_path_for_task(), COMPLETION_TARGET)

    def _get_tmp_path_for_run_dependencies(self) -> Path:
        return Path(self._get_tmp_path_for_task(), RUN_DEPENDENCIES)

    def get_log_path(self) -> Path:
        path = Path(self.get_output_path(), "logs")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_cache_path(self) -> Path:
        path = Path(build_config().output_directory, "cache")
        path.mkdir(parents=True, exist_ok=True)
        return path

    def register_required(self):
        pass

    def register_dependency(self, task: "BaseTask"):
        if self._task_state == TaskState.INIT:
            index = len(self._registered_tasks)
            self._registered_tasks.append(task)
            return RequiresTaskFuture(self, index)
        else:
            raise WrongTaskStateException(self._task_state, "register_dependency")

    def register_dependencies(self, tasks):
        if isinstance(tasks, dict):
            return {key: self.register_dependencies(task) for key, task in tasks.items()}
        elif isinstance(tasks, list):
            return [self.register_dependencies(task) for task in tasks]
        elif isinstance(tasks, BaseTask):
            return self.register_dependency(tasks)
        else:
            return tasks

    def get_values_from_futures(self, futures):
        if isinstance(futures, dict):
            return {key: self.get_values_from_futures(task) for key, task in futures.items()}
        elif isinstance(futures, list):
            return [self.get_values_from_futures(task) for task in futures]
        elif isinstance(futures, AbstractTaskFuture):
            return self.get_values_from_future(futures)
        else:
            return futures

    def get_values_from_future(self, future: AbstractTaskFuture) -> Union[Any, Dict[str, Any]]:
        if len(future.list_outputs()) == 1 and DEFAULT_RETURN_OBJECT_NAME in future.list_outputs():
            return future.get_output()
        else:
            return {future.get_output(key) for key in future.list_outputs()}

    def requires(self):
        return self._registered_tasks

    def output(self):
        return self._complete_target

    def run(self):
        try:
            self._task_state = TaskState.RUN
            task_generator = self.run_task()
            if task_generator is not None:
                yield from task_generator
            self._task_state = TaskState.NONE
            self.logger.info("Write complete_target")
            self._complete_target.write(self._registered_return_targets)
        except Exception as e:
            self.logger.exception("Exception in run: %s", e)
            raise e

    def run_task(self):
        raise AbstractMethodException()

    def run_dependencies(self, tasks) -> Generator["BaseTask", PickleTarget, Any]:
        if self._task_state == TaskState.RUN:
            self._register_run_dependencies(tasks)
            self._run_dependencies_target.write(self._run_dependencies_tasks)
            completion_targets = yield tasks
            task_futures = self._generate_run_task_furtures(completion_targets)
            return task_futures
        else:
            raise WrongTaskStateException(self._task_state, "run_dependency")

    def _register_run_dependencies(self, tasks):
        if isinstance(tasks, dict):
            for key, task in tasks.items():
                self._register_run_dependencies(task)
        elif isinstance(tasks, list):
            for task in tasks:
                self._register_run_dependencies(task)
        elif isinstance(tasks, BaseTask):
            self._run_dependencies_tasks.append(tasks)

    def _generate_run_task_furtures(self, completion_targets):
        if isinstance(completion_targets, dict):
            return {key: self._generate_run_task_furtures(task) for key, task in completion_targets.items()}
        elif isinstance(completion_targets, list):
            return [self._generate_run_task_furtures(task) for task in completion_targets]
        elif isinstance(completion_targets, PickleTarget):
            return RunTaskFuture(completion_targets)
        else:
            return completion_targets

    def return_object(self, object: Any, name: str = DEFAULT_RETURN_OBJECT_NAME):
        """Returns the object to the calling task. The object needs to be pickleable"""
        if self._task_state == TaskState.RUN:
            if name not in self._registered_return_targets:
                target = PickleTarget(self._get_tmp_path_for_returns(name))
                self._registered_return_targets[name] = target
                target.write(object)
            else:
                raise Exception(f"return target {name} already used")
        else:
            raise WrongTaskStateException(self._task_state, "return_target")

    def __repr__(self):
        """
        Build a task representation like `MyTask(param1=1.5, param2='5')`
        """
        params = self.get_params()
        param_values = self.get_param_values(params, [], self.param_kwargs)

        # Build up task id
        repr_parts = []
        param_objs = dict(params)
        for param_name, param_value in param_values:
            if param_objs[param_name].significant and \
                    param_objs[param_name].visibility == ParameterVisibility.PUBLIC:
                repr_parts.append('%s=%s' % (param_name, param_objs[param_name].serialize(param_value)))

        task_str = '{}({})'.format(self.task_id, ', '.join(repr_parts))

        return task_str

    def create_child_task_with_common_params(self, task_class, **kwargs):
        params = util.common_params(self, task_class)
        params["caller_output_path"] = self._extend_output_path()
        params.update(kwargs)
        return task_class(**params)

    def create_child_task(self, task_class, **kwargs):
        params = {}
        params["caller_output_path"] = self._extend_output_path()
        params.update(kwargs)
        return task_class(**params)

    def cleanup_child_task(self):
        if self._run_dependencies_target.exists():
            _run_dependencies_tasks_from_target = self._run_dependencies_target.read()
        else:
            _run_dependencies_tasks_from_target = []
        _run_dependencies_tasks = self._run_dependencies_tasks + _run_dependencies_tasks_from_target
        reversed_run_dependencies_task_list = list(_run_dependencies_tasks)
        reversed_run_dependencies_task_list.reverse()
        for task in reversed_run_dependencies_task_list:
            task.cleanup()

        reversed_registered_task_list = list(self._registered_tasks)
        reversed_registered_task_list.reverse()
        for task in reversed_registered_task_list:
            task.cleanup()

    def cleanup_task(self):
        pass

    def cleanup(self):
        self.logger.info("Cleaning up")
        if self._task_state != TaskState.CLEANUP and self._task_state != TaskState.CLEANED:
            self._task_state = TaskState.CLEANUP
            self.cleanup_child_task()
            self.cleanup_task()
            if self._get_tmp_path_for_task().exists():
                shutil.rmtree(self._get_tmp_path_for_task())
            self._task_state = TaskState.CLEANED
        self.logger.info("Cleanup finished")
