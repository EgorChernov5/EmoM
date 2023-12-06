from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image

from .utils import is_image, get_file_paths


def transform_target(target):
    CLASSNAME_TO_INT = dict(
        angry=0,
        disgust=1,
        fear=2,
        happy=3,
        neutral=4,
        sad=5,
        surprise=6
    )
    return CLASSNAME_TO_INT[target]


class EmoMDataset(Dataset):
    def __init__(self, classes_path: Path | str, image_paths: list[Path | str] = None, transform=None,
                 transform_target=transform_target):
        if image_paths is None:
            image_paths = get_file_paths(classes_path)

        self.image_paths = [image_path for image_path in image_paths if is_image(image_path)]
        self.transform = transform
        self.transform_target = transform_target

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = read_image(str(img_path))
        label = img_path.parent.name
        if self.transform:
            image = self.transform(image)

        if self.transform_target:
            label = self.transform_target(label)

        return image, label
