def check_extension(file_name: str):
    """
    The function returns whether the file is a photo.

    :param file_name:
    :return:
    """
    valid = ['bmp', 'dib', 'jpeg', 'jpg', 'jpe', 'jp2', 'png', 'webp', 'pbm', 'pgm', 'ppm',
             'pxm', 'pnm', 'pfm', 'sr', 'ras', 'tiff', 'tif', 'exr', 'hdr', 'pic']
    return file_name.split('.')[-1] in valid
