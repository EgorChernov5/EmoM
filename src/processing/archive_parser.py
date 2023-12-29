import zipfile
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import is_image, get_file_paths
from .filters import fltr_human_emo


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

    def parse_archive(self, archive_dir: Path | str, dataset_dir: Path | str = Path("raw/common"),
                      ffilter=is_image):
        """
        Parses the contents of an archive to a specified location.

        :param archive_dir: folder with archives.
        :param dataset_dir: folder where to save the contents of the archive.
        :param ffilter: a function that shows which objects to extract.
        """
        dataset_path = self.data_dir / dataset_dir
        dataset_path.mkdir(parents=True, exist_ok=True)
        temp_dp = dataset_path / "temp"
        temp_dp.mkdir(parents=True, exist_ok=True)
        # Extract
        print("Extract files to temp folder...")
        with zipfile.ZipFile(self.data_dir / archive_dir, 'r') as z:
            obj_paths = list(filter(ffilter, z.namelist()))
            z.extractall(path=temp_dp, members=obj_paths)

        # Rename
        class_folders = set()
        i = 1
        for obj_path in tqdm(obj_paths, desc="Renaming files"):
            obj_path = Path(obj_path)
            new_filename = f"{Path(dataset_path).name}_image_{i}.{obj_path.name.split('.')[-1]}"
            (temp_dp / obj_path).rename(temp_dp / obj_path.parent / new_filename)
            class_folders.add(obj_path.parent)
            i += 1

        for class_folder in tqdm(class_folders, desc="Renaming folders"):
            (temp_dp / class_folder).rename(temp_dp / class_folder.parent / self.FOLDER_TO_LABEL[class_folder.name.lower()])

        # Replace
        self.move_dataset(temp_dp, dataset_dir)

    def split_dataset(self, dataset_dir: Path | str, save_dir: Path | str = None, mtype: str = 'copy',
                      obj_paths: list[Path | str] = None, test_size: float = .2, shuffle=True, stratify=True):
        """
        Splits the processing and returns paths.

        :param dataset_dir: the relative path to dataset.
        :param save_dir: the relative path to the folder where we move the training and test objects. If None, then we do not move it.
        :param mtype: the type of movement. It can take values {'copy', 'replace'}.
        :param obj_paths: list of paths to objects in the dataset.
        :param test_size: should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
        :param shuffle: whether or not to shuffle the processing before splitting.
        :param stratify: if True, processing is split in a stratified fashion, using this as the class labels.
        :return: X_train, X_test
        """
        if obj_paths is None:
            obj_paths = self.get_file_paths(dataset_dir)

        stratify = [x.parent.name for x in obj_paths] if stratify else None
        X_train, X_test = train_test_split(obj_paths, test_size=test_size, random_state=self.random_state,
                                           shuffle=shuffle, stratify=stratify)
        if save_dir:
            self.move_dataset(dataset_dir, Path(save_dir) / 'train', X_train, mtype == 'copy')
            self.move_dataset(dataset_dir, Path(save_dir) / 'test', X_test, mtype == 'copy')

        return X_train, X_test

    def prepare_dataset(self, dataset_dir: Path | str, save_dir: Path | str, quarantine_dir: Path | str = None,
                        ffilter=fltr_human_emo, obj_paths: list[Path | str] = None) -> tuple[list[Path | str], list[Path | str]]:
        """
        The function that filters the processing, there are objects that are selected in the "ffilter" function.

        :param dataset_dir: the relative path to the dataset.
        :param save_dir: the path to the folder where to save.
        :param quarantine_dir: the path to the folder where the data that requires additional processing is saved.
        :param ffilter: link to the filtering function.
        :param obj_paths: list of paths to objects in the dataset.
        :return: save_paths, quarantine_paths
        """
        if obj_paths is None:
            obj_paths = self.get_file_paths(dataset_dir)

        save_paths, quarantine_paths = ffilter(obj_paths)
        self.move_dataset(dataset_dir, save_dir, save_paths, True, True)
        if quarantine_dir:
            self.move_dataset(dataset_dir, quarantine_dir, quarantine_paths, True, True)

        return save_paths, quarantine_paths

    def get_file_paths(self, dataset_dir: Path | str) -> list[Path]:
        """
        Returns a list of paths to objects in the dataset.

        :param dataset_dir: the relative path to the dataset.
        :return: obj_paths
        """
        dataset_path = self.data_dir / dataset_dir
        file_paths = get_file_paths(dataset_path)
        return [file_path for file_path in file_paths if is_image(file_path)]

    def move_dataset(self, src_dir: Path | str, dst_dir: Path | str, obj_paths: list[Path | str] = None,
                     copy_dataset=False, save_structure=False):
        """
        A function that moves processing.

        :param src_dir: the source folder.
        :param dst_dir: the destination folder.
        :param obj_paths: list of absolute or relative paths to objects in the dataset.
        :param copy_dataset: if True, copies the processing.
        :param save_structure: if True, keeps the folder structure.
        """
        # Replace
        if obj_paths is None:
            obj_paths = self.get_file_paths(src_dir)

        for obj_path in tqdm(obj_paths, desc="Moving"):
            obj_path = Path(str(obj_path).split(str(self.data_dir))[-1][1:])
            if save_structure:
                residual_path = Path(str(obj_path).split(str(Path(src_dir)))[-1][1:])
                print(residual_path)
                label_path = self.data_dir / dst_dir / residual_path.parent
            else:
                label_path = self.data_dir / dst_dir / obj_path.parent.name

            label_path.mkdir(parents=True, exist_ok=True)
            if copy_dataset:
                shutil.copy(self.data_dir / obj_path, label_path / obj_path.name)
            else:
                (self.data_dir / obj_path).replace(label_path / obj_path.name)

        # Delete empty dirs
        if not copy_dataset:
            self.delete_dataset(src_dir)

        print("Complete!")

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
