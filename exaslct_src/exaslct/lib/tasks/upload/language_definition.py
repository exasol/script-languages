from pathlib import Path

from jinja2 import Template


class LanguageDefinition():
    def __init__(self,
                 release_name: str,
                 flavor_path: str,
                 bucketfs_name: str,
                 bucket_name: str,
                 path_in_bucket: str,
                 add_missing_builtin: bool = False):
        self.path_in_bucket = path_in_bucket
        self.bucket_name = bucket_name
        self.bucketfs_name = bucketfs_name
        self.flavor_path = flavor_path
        self.release_name = release_name
        self.add_missing_builtin = add_missing_builtin

    def generate_definition(self):
        path_in_bucket = self.path_in_bucket
        if path_in_bucket != "" and not path_in_bucket.endswith("/"):
            path_in_bucket = path_in_bucket+"/"
        language_definition_path = Path(
            self.flavor_path, "flavor_base", "language_definition")
        with language_definition_path.open("r") as f:
            language_definition_template = f.read()
        template = Template(language_definition_template)
        language_definition = template.render(bucketfs_name=self.bucketfs_name,
                                              bucket_name=self.bucket_name,
                                              path_in_bucket=path_in_bucket,
                                              release_name=self.release_name)
        if self.add_missing_builtin:
            defined_aliases = [alias.split("=")[0]
                               for alias in language_definition.split(" ")]
            builtin_aliases = ["PYTHON", "JAVA", "R"]
            missing_aliases = set(builtin_aliases) - set(defined_aliases)
            additional_language_defintions = " ".join(
                alias+"=builtin_"+alias.lower() for alias in missing_aliases)
            language_definition = " ".join(
                [language_definition, additional_language_defintions])
        return language_definition.strip()

    def generate_alter_session(self):
        return f"""ALTER SESSION SET SCRIPT_LANGUAGES='{self.generate_definition()}';"""

    def generate_alter_system(self):
        return f"""ALTER SYSTEM SET SCRIPT_LANGUAGES='{self.generate_definition()}';"""
