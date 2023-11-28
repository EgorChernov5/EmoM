import zipfile
from pathlib import Path
import shutil

from utils import check_extension
from filters import fltr_human_emo


REPO_DIR = Path(__file__).parent.parent.parent


class EmoMParser:
    def __init__(self, data_dir: Path, folder_to_label: dict = None):
        self.data_dir = data_dir
        self.FOLDER_TO_LABEL = folder_to_label if folder_to_label else {
            'angry': 'angry',
            'anger': 'angry',
            'disgust': 'disgust',
            'disgusted': 'disgust',
            'fear': 'fear',
            'fearful': 'fear',
            'happy': 'happy',
            'happiness': 'happy',
            'neutral': 'neutral',
            'neutrality': 'neutral',
            'sad': 'sad',
            'sadness': 'sad',
            'surprise': 'surprise',
            'surprised': 'surprise',
        }
        self.image_id = 1

    def parse_archive(self, archive_dir: Path, dataset_dir: Path = None, ffilter=None, split=None, shuffle=True,
                      stratify=None):
        archive_path = self.data_dir / archive_dir
        dataset_path = self.data_dir / dataset_dir if dataset_dir else self.data_dir / "raw/common"
        dataset_path.mkdir(parents=True, exist_ok=True)
        temp_dp = dataset_path / "temp"
        temp_dp.mkdir()

        # Extract
        with zipfile.ZipFile(archive_path, 'r') as z:
            image_paths = list(filter(check_extension, z.namelist()))
            z.extractall(path=temp_dp, members=image_paths)

        # Replace
        for image_path in image_paths:
            source = temp_dp / image_path
            destination = dataset_path / self.FOLDER_TO_LABEL[source.parent.name.lower()]
            destination.mkdir(parents=True, exist_ok=True)
            image_name = f"{archive_dir.name.split('.')[0]}_image_{self.image_id}.{source.name.split('.')[-1]}"
            source.replace(destination / image_name)
            self.image_id += 1

        # Delete
        shutil.rmtree(temp_dp)
        self.image_id = 1

    def split_dataset(self):
        pass

    def merge_dataset(self):
        pass

    def delete_dataset(self):
        pass

    def replace_dataset(self):
        pass


# emom_parser = EmoMParser(REPO_DIR / "data")
# emom_parser.parse_archive(Path("archives/RatingOpenCVEmotionImages.zip"), Path("raw/RatingOpenCVEmotionImages"))
