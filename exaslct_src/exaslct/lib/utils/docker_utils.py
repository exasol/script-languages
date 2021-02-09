from typing import Callable, List


def find_images_by_tag(client, condition: Callable[[str], bool]) -> List:
    images = client.images.list()
    filter_images = [image for image in images
                     if image.tags is not None and len(image.tags) > 0 and
                     any([condition(tag) for tag in image.tags])]
    return filter_images
