import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_files(base_path, valid_ext=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(base_path):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind("."):].lower()

            if valid_ext is None or ext.endswith(valid_ext):
                image_path = os.path.join(rootDir, filename)
                yield image_path


def list_images(base_path, contains=None):
    return list_files(base_path, valid_ext=image_types, contains=contains)
