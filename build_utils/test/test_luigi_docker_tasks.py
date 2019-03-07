import unittest

import luigi

from build_utils.docker_pull_or_build_tasks import DockerPullOrBuildImageTask


class Task1(DockerPullOrBuildImageTask):
    pass

class Task2(DockerPullOrBuildImageTask):

    def requires(self):
        return [Task1(build_directories={"udfclient-deps": "resources/test-flavor/udfclient-deps",
                                         "ext": "../../ext"},
                      dockerfile="resources/test-flavor/udfclient-deps/Dockerfile",
                      image_name="scripting-language-container",
                      image_tag="test-flavor-udfclient-deps1",
                      ),
                Task1(build_directories={"udfclient-deps": "resources/test-flavor/udfclient-deps",
                                         "ext": "../../ext"},
                      dockerfile="resources/test-flavor/udfclient-deps/Dockerfile",
                      image_name="scripting-language-container",
                      image_tag="test-flavor-udfclient-deps1",
                       ),
                Task3()
                ]


class MyTestCase(unittest.TestCase):
    def test_something(self):
        task = Task2(build_directories={"udfclient-deps": "resources/test-flavor/udfclient-deps",
                                        "ext": "../../ext"},
                     dockerfile="resources/test-flavor/udfclient-deps/Dockerfile",
                     image_name="scripting-language-container",
                     image_tag="test-flavor-udfclient-deps2",
                     force_build=True, )
        luigi.build([task], workers=5, local_scheduler=True)


if __name__ == '__main__':
    unittest.main()
