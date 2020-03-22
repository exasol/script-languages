from exaslct_src.abstract_method_exception import AbstractMethodException


class BucketFSSyncChecker:
    def prepare_upload(self):
        raise AbstractMethodException()

    def wait_for_bucketfs_sync(self):
        raise AbstractMethodException()
