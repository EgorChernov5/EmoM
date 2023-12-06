from pathlib import Path


def is_image(file_name: Path | str):
    """
    The function returns whether the file is a photo.

    :param file_name:
    :return:
    """
    valid = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'pbm', 'pgm', 'ppm',
             'pxm', 'pnm', 'pfm', 'sr', 'ras', 'tiff', 'tif', 'exr', 'hdr', 'pic']
    return str(file_name).split('.')[-1] in valid


def get_file_paths(dataset_path: Path | str):
    return [obj_path for obj_path in Path(dataset_path).glob("**/*") if obj_path.is_file()]
