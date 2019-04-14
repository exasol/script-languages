import unittest

import docker


class MyTestCase(unittest.TestCase):
    def test_something(self):
        client = docker.from_env()
        try:
            generator = client.images.push(
                "tkilias/scripting-language-container",
                tag="test-flavor-udfclient_deps_J3ZULYRYTZ5DOIDQPFT5S664XUQF675EFZHZKBTNL3JGYQ5ZR2LQ",
                auth_config={"username":"",
                             "password":""},
                stream=True)
            for line in generator:
                print(line)
        finally:
            client.close()


if __name__ == '__main__':
    unittest.main()
