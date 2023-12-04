import zipfile
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

from utils import check_extension
from filters import fltr_human_emo


REPO_DIR = Path(__file__).parent.parent.parent


class EmoMParser:
    def __init__(self, data_dir: Path | str, folder_to_label: dict = None, random_state=42):
        self.data_dir = Path(data_dir)
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
        self.random_state = random_state

    def parse_archive(self, archive_dir: Path | str, dataset_dir: Path | str = Path("raw/common")):
        """
        Parses the contents of an archive to a specified location.

        :param archive_dir: folder with archives.
        :param dataset_dir: folder where to save the contents of the archive.
        """
        archive_path = self.data_dir / archive_dir
        dataset_path = self.data_dir / dataset_dir
        dataset_path.mkdir(parents=True, exist_ok=True)
        temp_dp = dataset_path / "temp"
        temp_dp.mkdir()
        # Extract
        with zipfile.ZipFile(archive_path, 'r') as z:
            obj_paths = list(filter(check_extension, z.namelist()))
            z.extractall(path=temp_dp, members=obj_paths)

        # Rename
        for i, obj_path in enumerate(obj_paths):
            obj_path = Path(obj_path)
            label_dir = obj_path.parent
            new_filename = f"{Path(archive_dir).name.split('.')[0]}_image_{i + 1}.{obj_path.name.split('.')[-1]}"
            (temp_dp / obj_path).rename(temp_dp / label_dir / new_filename)
            (temp_dp / label_dir).rename(temp_dp / label_dir.parent / self.FOLDER_TO_LABEL[label_dir.name.lower()])

        # Replace
        self.move_dataset(temp_dp, dataset_dir)

    def split_dataset(self, dataset_dir: Path | str, test_size: float = None, shuffle=True, stratify=True):
        """
        Splits the data and returns paths.

        :param dataset_dir: relative path to dataset.
        :param test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        :param shuffle: whether or not to shuffle the data before splitting.
        :param stratify: if True, data is split in a stratified fashion, using this as the class labels.
        :return: X_train, X_test
        """
        dataset_path = self.data_dir / dataset_dir
        objs = self.get_file_paths(dataset_path)
        labels = [x.parent.name for x in objs]
        stratify = labels if stratify else None
        return train_test_split(objs, test_size=test_size, random_state=self.random_state, shuffle=shuffle,
                                stratify=stratify)

    def prepare_dataset(self, ffilter, dataset_dir: Path | str):
        pass

    def get_file_paths(self, dataset_dir: Path | str) -> list[Path]:
        """
        Returns a list of paths to objects in the dataset.

        :param dataset_dir:
        :return: obj_paths
        """
        dataset_path = self.data_dir / dataset_dir
        return [obj_path for obj_path in dataset_path.glob("**/*") if obj_path.is_file()]

    def move_dataset(self, src_dir: Path | str, dst_dir: Path | str, obj_paths: list[Path | str] = None,
                     copy_dataset=False, save_structure=False):
        """
        A function that moves data.

        :param src_dir: the source folder.
        :param dst_dir: the destination folder.
        :param obj_paths: list of relative paths to objects in the dataset.
        :param copy_dataset: if True, copies the data.
        :param save_structure: if True, keeps the folder structure.
        """
        # Replace
        if obj_paths is None:
            obj_paths = self.get_file_paths(src_dir)

        for obj_path in obj_paths:
            if save_structure:
                residual_path = str(obj_path).split(str(src_dir))[-1][1:]
                label_path = self.data_dir / dst_dir / Path(residual_path).parent
            else:
                label_path = self.data_dir / dst_dir / Path(obj_path).parent.name

            label_path.mkdir(parents=True, exist_ok=True)
            if copy_dataset:
                shutil.copy(self.data_dir / obj_path, label_path / obj_path.name)
            else:
                (self.data_dir / obj_path).replace(label_path / obj_path.name)

        # Delete empty dirs
        if not copy_dataset:
            self.delete_dataset(src_dir)

    def delete_dataset(self, dataset_dir: Path | str):
        """
        The function deletes empty folders.

        :param dataset_dir: the folder in question.
        """
        dataset_path = self.data_dir / dataset_dir
        obj_dirs = [obj_dir for obj_dir in dataset_path.glob("**")]
        obj_dirs += [dataset_path]
        obj_dirs.sort(key=lambda obj_dir: len(str(obj_dir)), reverse=True)
        for obj_dir in obj_dirs:
            try:
                obj_dir.rmdir()
                print(f"LOG: {obj_dir} has been deleted...")
            except OSError:
                print(f"LOG: {obj_dir} is not empty...")
                continue
