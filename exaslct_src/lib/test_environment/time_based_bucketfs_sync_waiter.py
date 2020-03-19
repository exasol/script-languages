import time

from exaslct_src.lib.test_environment.bucketfs_sync_checker import BucketFSSyncChecker


class TimeBasedBucketFSSyncWaiter(BucketFSSyncChecker):

    def __init__(self,sync_time_estimation):
        self.sync_time_estimation = sync_time_estimation

    def prepare_upload(self):
        pass

    def wait_for_bucketfs_sync(self):
        time.sleep(self.sync_time_estimation)