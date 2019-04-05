import luigi


class build_config(luigi.Config):
    force_pull = luigi.BoolParameter(False)
    force_build = luigi.BoolParameter(False)
    log_build_context_content = luigi.BoolParameter(False)
    #keep_build_context = luigi.BoolParameter(False)
    temporary_base_directory = luigi.OptionalParameter(None)
    output_directory = luigi.Parameter(".build_ouput")