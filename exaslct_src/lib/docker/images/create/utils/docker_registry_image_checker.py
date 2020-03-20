import json
import multiprocessing as mp
from typing import Dict

import docker


class DockerRegistryImageChecker:
    def map(self, image: str, queue: mp.Queue):
        client = docker.from_env() # TODO replace by docker_config
        try:
            generator = client.api.pull(
                repository=image,
                stream=True,
            )
            for log_line in generator:
                log_line = log_line.decode("utf-8")
                log_line = log_line.strip('\r\n')
                json_output = json.loads(log_line)
                queue.put(json_output)
            queue.put(None)
        except Exception as e:
            queue.put(e)
        finally:
            client.close()

    def check(self, image: str):
        queue = mp.Queue()
        process = mp.Process(target=self.map, args=(image, queue))
        process.start()
        try:
            while True:
                value = queue.get()
                if isinstance(value, Exception):
                    raise value
                elif isinstance(value, Dict):
                    if "status" in value and value["status"].startswith("Pulling"):
                        return True
                    elif "errorDetail" in value:
                        # TODO logger debug
                        return False
                elif value is None:
                    return False
                else:
                    raise RuntimeError(f"Should not happen. Programming Error. Unknown value {value}")
        finally:
            process.terminate()
