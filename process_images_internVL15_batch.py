from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from typing import Tuple, List
from PIL import Image
import threading
import pickle
import torch
import queue
import copy
import time
import os

model_config = dict(
    num_beams=1,
    max_new_tokens=512,
    do_sample=False,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num
    )
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


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
            pixel_values = load_image(image, max_num=6).to(torch.bfloat16)
            self.image_queue.put((key, pixel_values))
            # Signals to queue job is done
            self.image_queue.task_done()


class InternVL15Descriptor:
    def __init__(
        self,
        db_path: str = "descriptions_internvl15.pkl",
        log_interval: int = 100,
        model_config: dict = model_config,
    ) -> None:
        """
        Initialize the InternVL15Descriptor with the specified database path, log interval and model configuration.

        Args:
            db_path (str): The path to the database file.
            log_interval (int): The interval at which to log the database statistics.
            model_config (dict): The configuration of the InstructBlip model.
        """

        self.db_path = db_path
        self.log_interval = log_interval
        self.model_config = model_config
        self.threaded_image_reader = ThreadedImageReader(max_size=30)
        self.threaded_image_reader.daemon = True
        self.threaded_image_reader.start()
        self.batch_size = 4

        self.image_count = 0
        self.load_db()
        self.load_model()

    def load_model(self) -> None:
        """
        Loads the instruct blip model and processor from Salesforce.
        The model is loaded in half precision mode on the GPU if available.
        """

        path = "OpenGVLab/Mini-InternVL-Chat-4B-V1-5"
        self.model = (
            AutoModel.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            .eval()
            .cuda()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
                self.db = {}
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

    def process(
        self, pixel_values_list: List[torch.Tensor], prompts: str
    ) -> Tuple[str, int]:
        """
        Process the image with the specified prompt.

        Args:
            image (Image.Image): The image to process.
            prompt (str): The prompt to use.

        Returns:
            Tuple[str, int]: The generated description and the number of generated tokens.
        """

        num_patches_list = [pixel_values.size(0) for pixel_values in pixel_values_list]
        pixel_values_list = torch.cat(
            [pixel_values.cuda() for pixel_values in pixel_values_list], dim=0
        )

        responses = self.model.batch_chat(
            self.tokenizer,
            pixel_values_list,
            num_patches_list=num_patches_list,
            questions=prompts,
            generation_config=self.model_config,
        )

        generated_tokens = 0
        for i in range(len(prompts)):
            generated_tokens += len(responses[i].split(" "))

        return responses, generated_tokens

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
            keys = []
            images = []
            prompts = []
            for i in range(self.batch_size):
                if self.threaded_image_reader.is_iq_empty():
                    break
                key, image = self.threaded_image_reader.image_queue.get()
                keys.append(key)
                images.append(image)
                prompts.append(prompt)

            description, num_tokens = self.process(images, prompts)
            self.image_count += self.batch_size
            generated_tokens += num_tokens
            self.db["data"][key] = description

            for i, key in enumerate(keys):
                self.db["data"][key] = description[i]

            for _ in range(self.batch_size):
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


prompt = "Provide an extended description on this image. Describe the pose (or stance) of the subject, how is the body arranged? Describe the outfit, the environment etc... Be thorough. [/INST]"
path = "/mnt/NAS/PoseDrawings/references_reworked_v2"


if __name__ == "__main__":
    db = InternVL15Descriptor()
    db.generate_descriptions(path, prompt)
