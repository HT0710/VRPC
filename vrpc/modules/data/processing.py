from typing import List, Tuple, Union
from pathlib import Path
import random
import psutil
import shutil
import os

from tqdm.contrib.concurrent import process_map
from rich import print
from PIL import Image
import numpy as np
import yaml
import cv2

from ..utils import workers_handler, tuple_handler


class ImageProcessing:
    @staticmethod
    def resize(
        image: np.ndarray, size: Union[int, List[int], Tuple[int]]
    ) -> np.ndarray:
        """
        Resize image to the specified dimensions.

        Args:
            image (np.ndarray): Input image as a NumPy array.
            size (Union[int, List[int], Tuple[int]]): A tuple specifying the target size (width, height).

        Returns:
            np.ndarray: A NumPy array representing the resized image.
        """
        return cv2.resize(image, tuple_handler(size, 2))

    @staticmethod
    def add_border(
        image: np.ndarray, border_color: Tuple | int = (0, 0, 0)
    ) -> np.ndarray:
        """
        Adds a border around a given video frame to make it a square frame.

        Args:
            video (np.ndarray): Input video frame as a NumPy array.
            border_color (Tuple|int): Color of the border. Defaults to (0, 0, 0) for black.

        Returns:
            np.ndarray: A NumPy array representing the video after add border.
        """
        img_h, img_w = image.shape[:2]
        target_size = max(img_h, img_w)

        border_v = (target_size - img_h) // 2
        border_h = (target_size - img_w) // 2

        return cv2.copyMakeBorder(
            image,
            border_v,
            border_v,
            border_h,
            border_h,
            cv2.BORDER_CONSTANT,
            border_color,
        )


class VideoProcessing:
    """
    Process included:

    Load: Load video and return list of frames.
    Sampling: Choose 1 frame every n frames.
    Truncating: Equally trim both head and tail if video length > max_frame.
    Padding: Pad black frame to end of video if video length < min_frame.
    Resize: Change size of a video.
    """

    @staticmethod
    def load(path: str) -> List[np.ndarray]:
        """
        Load a video file and return it as a NumPy array.

        Args:
            path (str): The file path to the video.

        Returns:
            List[np.ndarray]: A NumPy array representing the video frames.

        Raises:
            - FileExistsError: If the specified file does not exist.
            - RuntimeError: If the video file cannot be opened.
        """
        # Check path
        if not os.path.exists(path):
            raise FileExistsError("File not found!")
        # Load video
        video = cv2.VideoCapture(path)
        # Check video
        if not video.isOpened():
            raise RuntimeError("Could not open video file.")
        # Extract frames
        output = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for _, frame in iter(video.read, (False, None))
        ]
        video.release()
        return output

    @staticmethod
    def sampling(
        video: Union[np.ndarray, List[np.ndarray]], value: int
    ) -> List[np.ndarray]:
        """
        Perform frame sampling on a video.

        Args:
            video (Union[np.ndarray, List[np.ndarray]]): Input video as a NumPy array.
            value (int): The sampling value. If 0, no sampling is performed.

        Returns:
            List[np.ndarray]: A NumPy array representing the sampled video frames.
        """
        return video[::value]

    @staticmethod
    def truncating(
        video: Union[np.ndarray, List[np.ndarray]], max_frame: int
    ) -> List[np.ndarray]:
        """
        Truncate a video to a specified maximum number of frames.

        Args:
            video (Union[np.ndarray, List[np.ndarray]]): Input video as a NumPy array.
            max_frame (int): The maximum number of frames to retain.

        Returns:
            List[np.ndarray]: A NumPy array representing the truncated video frames.
        """
        middle_frame = len(video) // 2
        m = max_frame // 2
        r = max_frame % 2
        return video[middle_frame - m : middle_frame + m + r]

    @staticmethod
    def padding(
        video: Union[np.ndarray, List[np.ndarray]], min_frame: int
    ) -> List[np.ndarray]:
        """
        Pad a video with black frames to meet a minimum frame length.

        Args:
            video (Union[np.ndarray, List[np.ndarray]]): Input video as a NumPy array.
            min_frame (int): The desired minimum length of the video in frames.

        Returns:
            List[np.ndarray]: A NumPy array representing the video after padding.
        """
        zeros_array = np.zeros((min_frame, *np.array(video).shape[1:]), dtype=np.uint8)
        zeros_array[: len(video), ...] = video
        return zeros_array

    @staticmethod
    def resize(
        video: Union[np.ndarray, List[np.ndarray]],
        size: Union[int, List[int], Tuple[int]],
    ) -> List[np.ndarray]:
        """
        Resize each frame of video to the specified dimensions.

        Args:
            video (Union[np.ndarray, List[np.ndarray]]): Input video as a NumPy array.
            size (Union[int, List[int], Tuple[int]]): A tuple specifying the target size (width, height).

        Returns:
            List[np.ndarray]: A NumPy array representing the resized video.
        """
        return [cv2.resize(frame, tuple_handler(size, 2)) for frame in video]


class ImagePreparation:
    def __init__(
        self,
        save_folder: str = "./data",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        image_size: Tuple[int, int] | int = (224, 224),
        num_workers: int = 0,
        keep_temp: bool = False,
    ) -> None:
        """
        Process video into frames, applying customizable options, and store the result to the specified destination.

        Args:
            save_folder (str, optional): The folder where processed data will be saved. Default: "./data"
            train_val_test_split (tuple, optional): Split ratios for train, validation, and test data. Default: (0.7, 0.15, 0.15)
            image_size (tuple | int, optional): Size of the output frames. Default: (224, 224)
            sampling_value (int, optional): Choose 1 frame every n frames. Default: 0
            max_frame (int, optional): Equally trim both head and tail if video length > max_frame. Default: 0
            min_frame (int, optional): Pad black frame to end of video if video length < min_frame. Default: 0
            num_workers (int, optional): Number of workers to use for parallel processing. Default: 0
            keep_temp (bool, optional): Whether to keep temporary files. Default: False
        """
        self.save_folder = save_folder
        self.split_size = train_val_test_split
        self.size = image_size
        self.keep_temp = keep_temp
        self.workers = workers_handler(num_workers)
        self.extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]

    def __call__(self, data_path: str, save_name: str = None, remake=False) -> None:
        """
        Automatically process data and save it to the specified location.

        Args:
            data_path (str): The path to the data to be processed.
            save_name (str): Name for the saved data.
            remake (bool): Whether to recreate the data if it already exists.

        Returns:
            None
        """
        return self.auto(data_path, save_name, remake)

    def _split_data(self, class_folder: str) -> None:
        """
        Splits the data in the given class folder into train, validation, and test sets.

        Args:
            class_folder (str): The name of the class folder containing the data to be split.

        Returns:
            None
        """

        # Create new path train, val, test folder
        dst_paths = [
            os.path.join(self.save_path, folder, class_folder)
            for folder in ["train", "val", "test"]
        ]
        [os.makedirs(path, exist_ok=True) for path in dst_paths]

        # Get all video path in source folder
        class_path = os.path.join(self.data_path, class_folder)
        video_paths = [
            str(video)
            for ext in self.extensions
            for video in Path(class_path).rglob("*" + ext)
        ]

        # Shuffle data
        random.shuffle(video_paths)

        # Split into train, val, test chunk
        split_sizes = [round(len(video_paths) * size) for size in self.split_size]
        splited_video_paths = [
            video_paths[: split_sizes[0]],
            video_paths[split_sizes[0] : split_sizes[0] + split_sizes[1]],
            video_paths[split_sizes[0] + split_sizes[1] :],
        ]

        # Copy to new destination
        for video_paths, split_path in zip(splited_video_paths, dst_paths):
            for video_path in video_paths:
                dst_path = os.path.join(split_path, video_path.split("/")[-1])
                if not os.path.exists(dst_path):
                    shutil.copyfile(video_path, dst_path)

    def _save_config(self, path: str) -> None:
        """
        Save data processed information

        Args:
            path (str): Path to store the information file.

        Returns:
            None
        """
        with open(path, "w") as config:
            config.write(
                "# This folder containing processed data\n"
                "\n"
                f"# Number of images: {len([image for ext in self.extensions for image in Path(self.save_path).rglob('*' + ext)])}\n"
                f"# Number of classes: {len(os.listdir(self.data_path))}\n"
                "\n"
            )
            yaml.dump(
                {
                    "train_val_test_split": tuple(self.split_size),
                    "image_size": tuple(self.size),
                },
                config,
                default_flow_style=False,
            )

    def auto(
        self, data_path: str, save_name: str = False, remake: bool = False
    ) -> None:
        """
        Automatically process data and save it to the specified location.

        Args:
            data_path (str): The path to the data to be processed.
            save_name (str, optional): Name for the saved data. Default: False
            remake (bool, optional): Whether to recreate the data if it already exists. Default: False

        Returns:
            None
        """

        # Check if data path is existed
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        # Define some important variables
        self.data_path = data_path
        self.data_name = save_name if save_name else self.data_path.split("/")[-1]
        self.save_path = os.path.join(
            self.save_folder,
            self.data_name
            + (
                "_x"
                if os.path.exists(os.path.join(self.save_folder, self.data_name))
                else ""
            ),
        )

        if os.path.exists(self.save_path):
            if remake:
                shutil.rmtree(self.save_path)
            else:
                print("[bold]Processing data is already existed.[/]")
                return

        # Process summary
        print(f"\n[bold]Summary:[/]")
        print(f"  Remake: {remake}")
        print(f"  Number of workers: {self.workers}")
        print(f"  Data path: [green]{self.data_path}[/]")
        print(f"  Save path: [green]{self.save_path}[/]")

        # Calcute chunksize base on cpu parallel power
        benchmark = lambda x: max(
            1, round(len(x) / (self.workers * psutil.cpu_freq().max / 1000) / 4)
        )

        # Split data
        print("\n[bold][yellow]Splitting data...[/][/]")
        class_folders = os.listdir(self.data_path)
        process_map(
            self._split_data,
            class_folders,
            max_workers=self.workers,
            chunksize=benchmark(class_folders),
        )

        # Save configuration
        print("\n[bold][yellow]Saving config...[/][/]")
        self._save_config(os.path.join(self.save_path, "config.yaml"))

        print("\n[bold][green]Processing data complete.[/][/]")


class VideoPreparation:
    def __init__(
        self,
        save_folder: str = "./data",
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        image_size: Tuple[int, int] | int = (224, 224),
        sampling_value: int = 0,
        max_frame: int = 0,
        min_frame: int = 0,
        num_workers: int = 0,
        keep_temp: bool = False,
    ) -> None:
        """
        Process video into frames, applying customizable options, and store the result to the specified destination.

        Args:
            save_folder (str, optional): The folder where processed data will be saved. Default: "./data"
            train_val_test_split (tuple, optional): Split ratios for train, validation, and test data. Default: (0.7, 0.15, 0.15)
            image_size (tuple | int, optional): Size of the output frames. Default: (224, 224)
            sampling_value (int, optional): Choose 1 frame every n frames. Default: 0
            max_frame (int, optional): Equally trim both head and tail if video length > max_frame. Default: 0
            min_frame (int, optional): Pad black frame to end of video if video length < min_frame. Default: 0
            num_workers (int, optional): Number of workers to use for parallel processing. Default: 0
            keep_temp (bool, optional): Whether to keep temporary files. Default: False
        """
        self.save_folder = save_folder
        self.split_size = train_val_test_split
        self.size = image_size
        self.sampling = sampling_value
        self.max_frame = max_frame
        self.min_frame = min_frame
        self.keep_temp = keep_temp
        self.workers = workers_handler(num_workers)
        self.extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv", ".mpg"]

    def __call__(self, data_path: str, save_name: str = None, remake=False) -> None:
        """
        Automatically process data and save it to the specified location.

        Args:
            data_path (str): The path to the data to be processed.
            save_name (str): Name for the saved data.
            remake (bool): Whether to recreate the data if it already exists.

        Returns:
            None
        """
        return self.auto(data_path, save_name, remake)

    def _split_data(self, class_folder: str) -> None:
        """
        Splits the data in the given class folder into train, validation, and test sets.

        Args:
            class_folder (str): The name of the class folder containing the data to be split.

        Returns:
            None
        """

        # Create new path train, val, test folder
        dst_paths = [
            os.path.join(self.temp_path, folder, class_folder)
            for folder in ["train", "val", "test"]
        ]
        [os.makedirs(path, exist_ok=True) for path in dst_paths]

        # Get all video path in source folder
        class_path = os.path.join(self.data_path, class_folder)
        video_paths = [
            str(video)
            for ext in self.extensions
            for video in Path(class_path).rglob("*" + ext)
        ]

        # Shuffle data
        random.shuffle(video_paths)

        # Split into train, val, test chunk
        split_sizes = [round(len(video_paths) * size) for size in self.split_size]
        splited_video_paths = [
            video_paths[: split_sizes[0]],
            video_paths[split_sizes[0] : split_sizes[0] + split_sizes[1]],
            video_paths[split_sizes[0] + split_sizes[1] :],
        ]

        # Copy to new destination
        for video_paths, split_path in zip(splited_video_paths, dst_paths):
            for video_path in video_paths:
                dst_path = os.path.join(split_path, video_path.split("/")[-1])
                if not os.path.exists(dst_path):
                    shutil.copyfile(video_path, dst_path)

    def __process_video(self, path: str) -> List[np.ndarray]:
        """
        Auto process the video.

        Args:
            path (str): Path to video to be processed.

        Returns:
            List[np.ndarray]: List of frames of the video.
        """
        video = VideoProcessing.load(path)
        if self.sampling:
            video = VideoProcessing.sampling(np.array(video), self.sampling)
        if self.max_frame:
            video = VideoProcessing.truncating(np.array(video), self.max_frame)
        if self.min_frame:
            video = VideoProcessing.padding(np.array(video), self.min_frame)
        if self.size:
            video = VideoProcessing.resize(np.array(video), self.size)
        return video

    def _frame_generate(self, path: str) -> None:
        """
        Generate and save individual frames from a video file.

        Args:
            path (str): The path to the video file.

        Returns:
            None
        """

        # Get file name
        file_name = path.split("/")[-1][:-4]

        # Make destination path
        dst_path = "/".join(
            path.replace(self.temp_path, self.save_path).split("/")[:-1]
        )

        # Create it if not existed
        os.makedirs(dst_path, exist_ok=True)

        # Process the video
        video = self.__process_video(path)

        # Save each frame to the destination
        for i, frame in enumerate(video):
            save_path = os.path.join(dst_path, f"{file_name}_{i}.jpg")
            if not os.path.exists(save_path):
                img = Image.fromarray(frame)
                img.save(save_path)

    def _save_config(self, path: str) -> None:
        """
        Save data processed information

        Args:
            path (str): Path to store the information file.

        Returns:
            None
        """
        with open(path, "w") as config:
            config.write(
                "# This folder containing processed data\n"
                "\n"
                f"# Number of images: {len([image for image in Path(self.save_path).rglob('*.jpg')])}\n"
                f"# Number of classes: {len(os.listdir(self.data_path))}\n"
                "\n"
            )
            yaml.dump(
                {
                    "train_val_test_split": tuple(self.split_size),
                    "sampling_value": self.sampling,
                    "max_frame": self.max_frame,
                    "min_frame": self.min_frame,
                    "image_size": tuple(self.size),
                },
                config,
                default_flow_style=False,
            )

    def auto(self, data_path: str, save_name: str = False, remake=False) -> None:
        """
        Automatically process data and save it to the specified location.

        Args:
            data_path (str): The path to the data to be processed.
            save_name (str, optional): Name for the saved data. Default: False
            remake (bool, optional): Whether to recreate the data if it already exists. Default: False

        Returns:
            None
        """

        # Check if data path is existed
        if not os.path.exists(data_path):
            raise FileNotFoundError(data_path)

        # Define some important variables
        self.data_path = data_path
        self.data_name = save_name if save_name else self.data_path.split("/")[-1]
        self.temp_path = os.path.join(self.save_folder, self.data_name + "_tmp")
        self.save_path = os.path.join(
            self.save_folder,
            self.data_name
            + (
                "_x"
                if os.path.exists(os.path.join(self.save_folder, self.data_name))
                else ""
            ),
        )

        if os.path.exists(self.save_path):
            if remake:
                shutil.rmtree(self.save_path)
            else:
                print("[bold]Processing data is already existed.[/]")
                return

        # Process summary
        print(f"\n[bold]Summary:[/]")
        print(f"  Remake: {remake}")
        print(f"  Number of workers: {self.workers}")
        print(f"  Data path: [green]{self.data_path}[/]")
        print(f"  Save path: [green]{self.save_path}[/]")

        # Calcute chunksize base on cpu parallel power
        benchmark = lambda x: max(
            1, round(len(x) / (self.workers * psutil.cpu_freq().max / 1000) / 4)
        )

        # Split data
        print("\n[bold][yellow]Splitting data...[/][/]")
        class_folders = os.listdir(self.data_path)
        process_map(
            self._split_data,
            class_folders,
            max_workers=self.workers,
            chunksize=benchmark(class_folders),
        )

        # Generate data
        print("\n[bold][yellow]Generating data...[/][/]")
        video_paths = [
            str(video)
            for ext in self.extensions
            for video in Path(self.temp_path).rglob("*" + ext)
        ]
        process_map(
            self._frame_generate,
            video_paths,
            max_workers=self.workers,
            chunksize=benchmark(video_paths),
        )

        # Save configuration
        print("\n[bold][yellow]Saving config...[/][/]")
        self._save_config(os.path.join(self.save_path, "config.yaml"))

        # Remove temp folder
        shutil.rmtree(self.temp_path) if not self.keep_temp else None
        print("\n[bold][green]Processing data complete.[/][/]")
