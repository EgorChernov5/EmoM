import cv2 as cv
from rmn import RMN
from pathlib import Path
from tqdm import tqdm


def fltr_human_emo(image_paths: list[Path | str]) -> tuple[list[Path | str], list[Path | str]]:
    """
    if the image does not pass filtering, it is quarantined.

    :param image_paths: list of paths to objects in the dataset.
    :return: save_paths, quarantine_paths
    """
    # Emotion classification model
    m = RMN()
    # Processing images
    save_paths = []
    quarantine_paths = []
    for image_path in tqdm(image_paths):
        img = cv.imread(str(image_path))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = m.detect_emotion_for_single_frame(img)
        for result in results:
            if result['emo_label'] == image_path.parent.name and result['emo_proba'] > .6:
                save_paths += [image_path]
                break

        if image_path not in save_paths:
            quarantine_paths += [image_path]

    return save_paths, quarantine_paths
