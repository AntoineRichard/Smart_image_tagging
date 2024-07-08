from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from typing import Tuple, List
from PIL import Image
import threading
import pickle
import torch
import queue
import copy
import time
import os

model_config = {
    "do_sample": False,
    "num_beams": 5,
    "max_length": 512,
    "min_length": 1,
    "top_p": 0.9,
    "repetition_penalty": 1.5,
    "length_penalty": 1.0,
    "temperature": 1,
}


class ThreadedImageReader(threading.Thread):
    def __init__(self, max_size: int = 100) -> None:
        """
        Initialize the ThreadedImageReader with the specified maximum size of the image queue.

        Args:
            max_size (int): The maximum size of the image queue.
        """

        threading.Thread.__init__(self)
        self.image_queue = queue.Queue(maxsize=max_size)
        self.path_queue = queue.Queue(maxsize=max_size)

    def is_iq_empty(self) -> bool:
        """
        Check if the image queue is empty.

        Returns:
            bool: True if the image queue is empty, False otherwise.
        """

        return self.image_queue.empty()

    def is_iq_full(self) -> bool:
        """
        Check if the image queue is full.

        Returns:
            bool: True if the image queue is full, False otherwise.
        """

        return self.image_queue.full()

    def is_pq_empty(self) -> bool:
        """
        Check if the path queue is empty.

        Returns:
            bool: True if the path queue is empty, False otherwise.
        """

        return self.path_queue.empty()

    def is_pq_full(self) -> bool:
        """
        Check if the path queue is full.

        Returns:
            bool: True if the path queue is full, False otherwise.
        """

        return self.path_queue.full()

    def run(self) -> None:
        """
        Run the ThreadedImageReader thread.
        Fetches image pathes from the path queue and puts them into the image queue.
        """

        while True:
            key, image_path = self.path_queue.get()
            image = Image.open(image_path).convert("RGB")
            self.image_queue.put((key, image))
            # Signals to queue job is done
            self.image_queue.task_done()


class InstructBlipDescriptor:
    def __init__(
        self,
        db_path: str = "descriptions_instruct_blip.pkl",
        log_interval: int = 100,
        model_config: dict = model_config,
    ) -> None:
        """
        Initialize the InstructBlipDescriptor with the specified database path, log interval and model configuration.

        Args:
            db_path (str): The path to the database file.
            log_interval (int): The interval at which to log the database statistics.
            model_config (dict): The configuration of the InstructBlip model.
        """

        self.db_path = db_path
        self.log_interval = log_interval
        self.model_config = model_config
        self.threaded_image_reader = ThreadedImageReader(max_size=10)
        self.threaded_image_reader.daemon = True
        self.threaded_image_reader.start()

        self.image_count = 0
        self.load_db()
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the instruct blip model and processor from Salesforce.
        The model is loaded in half precision mode on the GPU if available.
        """

        self.model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b", torch_dtype=torch.float16
        )
        self.processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-vicuna-7b", torch_dtype=torch.float16
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def load_db(self) -> None:
        """
        Load the database from the specified path.
        """

        if os.path.exists(self.db_path):
            print("Loading already existing database.")
            with open(self.db_path, "rb") as f:
                self.db = pickle.load(f)
            # Compat with old dbs without statistics
            if "data" not in self.db.keys():
                print("Old database format detected. Updating to new format.")
                db_old = copy.copy(self.db)
                self.db["data"] = db_old
                self.db["statistics"] = {}
                self.db["statistics"]["processed_images"] = len(self.db["data"])
                self.db["statistics"]["generated_tokens"] = 0
                self.db["statistics"]["run_time"] = 0
                self.db["statistics"]["avg_it_per_second"] = 0
                self.db["statistics"]["avg_tokens_per_second"] = 0
            self.image_count = self.db["statistics"]["processed_images"]
            print("Database loaded.")
            print("Current database statistics:")
            print("Processed images: " + str(self.db["statistics"]["processed_images"]))
            print("Generated tokens: " + str(self.db["statistics"]["generated_tokens"]))
            print("Run time: " + str(self.db["statistics"]["run_time"]))
            print(
                "Average iterations per second: "
                + str(self.db["statistics"]["avg_it_per_second"])
            )
            print(
                "Average tokens per second: "
                + str(self.db["statistics"]["avg_tokens_per_second"])
            )
        else:
            print("No database found. Creating new one.")
            self.db = {}
            self.db["data"] = {}
            self.db["statistics"] = {}
            self.db["statistics"]["processed_images"] = 0
            self.db["statistics"]["generated_tokens"] = 0
            self.db["statistics"]["run_time"] = 0
            self.db["statistics"]["avg_it_per_second"] = 0
            self.db["statistics"]["avg_tokens_per_second"] = 0

    def save_db(self) -> None:
        """
        Save the database to the specified path.
        """

        with open(self.db_path, "wb") as f:
            pickle.dump(self.db, f)

    def process(self, image: Image.Image, prompt: str) -> Tuple[str, int]:
        """
        Process the image with the specified prompt.

        Args:
            image (Image.Image): The image to process.
            prompt (str): The prompt to use.

        Returns:
            Tuple[str, int]: The generated description and the number of generated tokens.
        """

        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

        outputs = self.model.generate(
            **inputs,
            **self.model_config,
        )
        generated_tokens = outputs[0].shape[0]
        return (
            self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip(),
            generated_tokens,
        )

    def update_statistics(
        self, last_ckpt: float, generated_tokens: int
    ) -> Tuple[float, int]:
        """
        Update the database statistics with the specified values.

        Args:
            last_ckpt (float): The last checkpoint time.
            generated_tokens (int): The number of generated tokens.

        Returns:
            Tuple[float, int]: The current checkpoint time and the number of generated tokens.
        """

        ckpt = time.time()
        self.db["statistics"]["processed_images"] = self.image_count
        self.db["statistics"]["generated_tokens"] += generated_tokens
        self.db["statistics"]["run_time"] += ckpt - last_ckpt
        self.db["statistics"]["avg_it_per_second"] = (
            self.db["statistics"]["processed_images"]
            / self.db["statistics"]["run_time"]
        )
        self.db["statistics"]["avg_tokens_per_second"] = (
            self.db["statistics"]["generated_tokens"]
            / self.db["statistics"]["run_time"]
        )
        generated_tokens = 0
        return copy.copy(ckpt), generated_tokens

    def log(self) -> None:
        """
        Log the database statistics.
        """

        print(
            "processed "
            + str(self.db["statistics"]["processed_images"])
            + " images in "
            + str(self.db["statistics"]["run_time"])
            + " seconds."
        )
        print(
            "Average iterations per second: "
            + str(self.db["statistics"]["avg_it_per_second"])
        )
        print(
            "Average tokens per second: "
            + str(self.db["statistics"]["avg_tokens_per_second"])
        )

    @staticmethod
    def make_path_generator(path_list: Tuple[str, str]) -> iter:
        """
        Make a path generator from the specified path list.

        Args:
            path_list (Tuple[str, str]): The path list to generate paths from.

        Returns:
            iter: The path generator.
        """

        for path in path_list:
            yield path

    def collect_paths(self, path: str) -> List[str]:
        """
        Collect all image paths from the specified path.

        Args:
            path (str): The path to collect the image paths from.

        Returns:
            List[str]: The collected image paths.
        """

        # Collect all image paths
        start = time.time()
        collected_paths = []
        for root, dirs, files in os.walk(path):
            for key in files:
                file_path = os.path.join(root, key)
                if key in self.db["data"]:
                    continue
                collected_paths.append((key, file_path))

        end = time.time()
        print("Collected all paths in " + str(end - start) + " seconds.")
        return collected_paths

    def prefetch_images(self, path_generator: iter) -> None:
        """
        Prefetch images from the specified path generator.

        Args:
            path_generator (iter): The path generator to prefetch images from.
        """

        start = time.time()
        # Prefetch images
        while not self.threaded_image_reader.is_pq_full():
            self.threaded_image_reader.path_queue.put(path_generator.__next__())
        while not self.threaded_image_reader.is_iq_full():
            time.sleep(0.1)
        end = time.time()
        print(
            "Prefetched "
            + str(self.threaded_image_reader.image_queue.qsize())
            + " images in "
            + str(end - start)
            + " seconds."
        )

    def generate_descriptions(self, path: str, prompt: str) -> None:
        """
        Generate descriptions for the images in the specified path with the specified prompt.

        Args:
            path (str): The path to the images.
            prompt (str): The prompt to use.
        """

        generated_tokens = 0

        collected_paths = self.collect_paths(path)
        path_generator = self.make_path_generator(collected_paths)
        self.prefetch_images(path_generator)

        last_ckpt = time.time()
        while True:
            key, image = self.threaded_image_reader.image_queue.get()
            description, num_tokens = self.process(image, prompt)
            self.image_count += 1
            generated_tokens += num_tokens
            self.db["data"][key] = description

            try:
                self.threaded_image_reader.path_queue.put(path_generator.__next__())
            except StopIteration:
                print("Generator exhausted. All images are in the queue.")

            if (self.image_count % self.log_interval) == 0 and (self.image_count != 0):
                last_ckpt, generated_tokens = self.update_statistics(
                    last_ckpt, generated_tokens
                )
                self.log()
                self.save_db()

            if (
                self.threaded_image_reader.is_pq_empty()
                and self.threaded_image_reader.is_iq_empty()
            ):
                break

        self.update_statistics(last_ckpt, generated_tokens)
        self.save_db()


if __name__ == "__main__":
    prompt = "Provide an extended description on this image. Describe the pose (or stance) of the subject, how is the body arranged? Describe the outfit, the environment etc... Be thorough."
    path = "/mnt/NAS/PoseDrawings/references_reworked_v2"

    db = InstructBlipDescriptor()
    db.generate_descriptions(path, prompt)
