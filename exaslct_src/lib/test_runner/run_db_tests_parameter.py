import luigi

from exaslct_src.lib.base.json_pickle_parameter import JsonPickleParameter
from exaslct_src.lib.data.environment_info import EnvironmentInfo


class GeneralRunDBTestParameter():
    test_restrictions = luigi.ListParameter([])
    test_environment_vars = luigi.DictParameter({"TRAVIS": ""}, significant=False)
    test_log_level = luigi.Parameter("critical", significant=False)


class ActualRunDBTestParameter(GeneralRunDBTestParameter):
    release_goal = luigi.Parameter()
    language_definition = luigi.Parameter(significant=False)
    test_environment_info = JsonPickleParameter(EnvironmentInfo, significant=False)  # type: EnvironmentInfo


class RunDBTestParameter(ActualRunDBTestParameter):
    language = luigi.OptionalParameter()


class RunDBGenericLanguageTestParameter(GeneralRunDBTestParameter):
    generic_language_tests = luigi.ListParameter([])


class RunDBLanguageTestParameter(GeneralRunDBTestParameter):
    languages = luigi.ListParameter([None])


class RunDBTestFolderParameter(RunDBLanguageTestParameter):
    test_folders = luigi.ListParameter([])


class RunDBTestFilesParameter(RunDBLanguageTestParameter):
    test_files = luigi.ListParameter([])


class RunDBTestsInTestConfigParameter(RunDBGenericLanguageTestParameter,
                                      RunDBTestFolderParameter,
                                      RunDBTestFilesParameter):
    pass
