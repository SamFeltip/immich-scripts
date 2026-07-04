import os

from scrollingLogDisplay import ScrollingLogDisplay

# Keep TensorFlow's own C++ / retracing chatter out of the console before
# tensorflow is even imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import logging
import threading
from collections import deque
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from dotenv import load_dotenv

# Silence tensorflow's own logger (this is what was emitting the
# "N out of the last N calls ... triggered tf.function retracing" spam).
tf.get_logger().setLevel("ERROR")

# =======================
# ENV / CONFIG
# =======================

load_dotenv()


def _get_env(name: str, default: Optional[str] = None) -> str:
    """
    Fetch environment variable.

    :param name: Variable name
    :param default: Optional default
    :return: Value
    :raises ValueError: If missing and no default provided
    """
    value = os.getenv(name, default)
    if value is None:
        raise ValueError(f"Missing env var: {name}")
    return value


def _get_int(name: str, default: int) -> int:
    """
    Fetch integer environment variable.

    :param name: Variable name
    :param default: Default value
    :return: Parsed int
    """
    return int(os.getenv(name, default))


IMMICH_BASE_URL: str = _get_env("IMMICH_BASE_URL")
API_KEY: str = _get_env("API_KEY")
NIMA_MODEL_PATH: str = _get_env("NIMA_MODEL_PATH", "./nima_model.h5")

MAX_WORKERS: int = _get_int("MAX_WORKERS", 6)
BATCH_SIZE: int = _get_int("BATCH_SIZE", 10)

HEADERS: Dict[str, str] = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json",
}

# =======================
# LOGGING
# =======================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)




# =======================
# MODEL
# =======================


def load_model() -> tf.keras.Model:
    """
    Load the NIMA model.

    The .h5 file shipped with the neural-image-assessment repo contains
    weights only (no architecture/config), so we build the MobileNet-based
    NIMA architecture ourselves and load the weights into it.

    :return: TensorFlow model
    """
    logger.info("Loading NIMA model...")

    base_model = tf.keras.applications.MobileNet(
        input_shape=(224, 224, 3),
        include_top=False,
        pooling="avg",
        weights=None,
    )
    x = tf.keras.layers.Dropout(0.75)(base_model.output)
    x = tf.keras.layers.Dense(10, activation="softmax")(x)
    model = tf.keras.Model(base_model.input, x)

    model.load_weights(NIMA_MODEL_PATH)
    logger.info("Model loaded.")
    return model


def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for inference.

    :param image_bytes: Raw image bytes
    :return: Image tensor
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_score(model: tf.keras.Model, tensor: np.ndarray) -> float:
    """
    Predict NIMA aesthetic score.

    :param model: NIMA model
    :param tensor: Image tensor
    :return: Score (1-10)
    """
    preds = model.predict(tensor, verbose=0)[0]
    scores = np.arange(1, 11)
    return float(np.sum(preds * scores))


def to_stars(score: float) -> int:
    """
    Convert NIMA score to 1-5 stars.

    :param score: Score (1-10)
    :return: Stars (1-5)
    """
    return int(np.clip(round(score / 2), 1, 5))


def apply_people_boost(stars: int, people_count: int) -> int:
    """
    Boost a star rating based on how many identified people are in the shot.

    :param stars: Base star rating (1-5)
    :param people_count: Number of identified (named) people in the asset
    :return: Boosted star rating, clamped to 5
    """
    if people_count > 4:
        boost = 2
    elif people_count >= 1:
        boost = 1
    else:
        boost = 0
    return int(np.clip(stars + boost, 1, 5))


# =======================
# IMMICH API
# =======================


def get_assets(album_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all assets in album.

    :param album_id: Album ID
    :return: List of assets
    """
    url = f"{IMMICH_BASE_URL}/api/albums/{album_id}"
    logger.info(f"Fetching assets from album {album_id}")

    res = requests.get(url, headers=HEADERS)
    res.raise_for_status()

    assets = res.json().get("assets", [])
    logger.info(f"Found {len(assets)} assets")
    return assets


def download_thumbnail(asset_id: str) -> Optional[bytes]:
    """
    Download asset thumbnail.

    :param asset_id: Asset ID
    :return: Image bytes or None
    """
    url = f"{IMMICH_BASE_URL}/api/assets/{asset_id}/thumbnail"
    res = requests.get(url, headers=HEADERS)

    if res.status_code != 200:
        logger.warning(f"Thumbnail failed: {asset_id}")
        return None

    return res.content


def get_people_count(asset_id: str) -> int:
    """
    Fetch the number of identified (named) people in an asset.

    Immich's asset-detail endpoint returns a `people` array containing only
    faces that have been assigned to a named person; unrecognized/unassigned
    faces are excluded, which is what we want for this boost.

    :param asset_id: Asset ID
    :return: Count of identified people, 0 on failure
    """
    url = f"{IMMICH_BASE_URL}/api/assets/{asset_id}"
    res = requests.get(url, headers=HEADERS)

    if res.status_code != 200:
        logger.warning(f"Could not fetch people info for {asset_id}")
        return 0

    people = res.json().get("people", [])
    return len(people)


def update_rating(asset_id: str, rating: int, dryrun: bool) -> None:
    """
    Update asset rating in Immich.

    :param asset_id: Asset ID
    :param rating: Star rating
    :param dryrun: Perform a dry run without updating ratings
    """
    url = f"{IMMICH_BASE_URL}/api/assets/{asset_id}"
    if dryrun:
        return

    res = requests.put(url, headers=HEADERS, json={"rating": rating})

    if res.status_code not in (200, 204):
        logger.error(f"Failed to update {asset_id}: {res.text}")


# =======================
# WORKER
# =======================


def process_asset(
    asset: Dict[str, Any],
    model: tf.keras.Model,
    overwrite: bool,
    dryrun: bool,
    display: ScrollingLogDisplay,
) -> None:
    """
    Process a single asset.

    :param asset: Asset dict
    :param model: NIMA model
    :param overwrite: Overwrite existing ratings
    :param dryrun: Perform a dry run without updating ratings
    :param display: Shared scrolling log/progress display
    """
    asset_id: Optional[str] = asset.get("id")
    rating: Optional[int] = asset.get("rating")

    if not asset_id:
        display.advance()
        return

    if rating is not None and not overwrite:
        display.advance()
        return

    img = download_thumbnail(asset_id)
    if not img:
        display.advance()
        return

    try:
        tensor = preprocess(img)
        score = predict_score(model, tensor)
        stars = to_stars(score)

        people_count = get_people_count(asset_id)
        final_stars = stars

        # final_stars = apply_people_boost(stars, people_count)

        if people_count > 0:
            display.log(
                f"{asset_id} -> {score:.2f} ({stars}★) "
                f"+{final_stars - stars} for {people_count} people = {final_stars}★"
            )
        else:
            display.log(f"{asset_id} -> {score:.2f} ({final_stars}★)")

        update_rating(asset_id, final_stars, dryrun)

    except Exception as e:
        display.log(f"Error processing {asset_id}: {e}")
        logger.exception(f"Error processing {asset_id}: {e}")
    finally:
        display.advance()


# =======================
# MAIN
# =======================


def run(album_id: str, overwrite: bool, dryrun: bool) -> None:
    """
    Run full pipeline.

    :param album_id: Album ID
    :param overwrite: Overwrite existing ratings
    :param dryrun: Perform a dry run without updating ratings
    """
    logger.info("Starting run")
    logger.info(f"Album: {album_id}")
    logger.info(f"Workers: {MAX_WORKERS}, Batch: {BATCH_SIZE}")

    model = load_model()
    assets = get_assets(album_id)

    display = ScrollingLogDisplay(total=len(assets), desc="Rating assets")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for i in range(0, len(assets), BATCH_SIZE):
            batch = assets[i:i + BATCH_SIZE]

            for asset in batch:
                futures.append(
                    executor.submit(process_asset, asset, model, overwrite, dryrun, display)
                )

        for _ in as_completed(futures):
            pass

    display.close()
    logger.info("Completed.")


# =======================
# ENTRY
# =======================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rate Immich album with NIMA")
    parser.add_argument("album_id", help="Immich album ID")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ratings"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Perform a dry run without updating ratings"
    )

    args = parser.parse_args()

    run(album_id=args.album_id, overwrite=args.overwrite, dryrun=args.dryrun)