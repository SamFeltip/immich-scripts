import os
import argparse
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import requests
import numpy as np
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from dotenv import load_dotenv

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

    :return: TensorFlow model
    """
    logger.info("Loading NIMA model...")
    model = tf.keras.models.load_model(NIMA_MODEL_PATH, compile=False)
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
    :return: Score (1–10)
    """
    preds = model.predict(tensor, verbose=0)[0]
    scores = np.arange(1, 11)
    return float(np.sum(preds * scores))


def to_stars(score: float) -> int:
    """
    Convert NIMA score to 1–5 stars.

    :param score: Score (1–10)
    :return: Stars (1–5)
    """
    return int(np.clip(round(score / 2), 1, 5))


# =======================
# IMMICH API
# =======================


def get_assets(album_id: str) -> List[Dict[str, Any]]:
    """
    Fetch all assets in album.

    :param album_id: Album ID
    :return: List of assets
    """
    url = f"{IMMICH_BASE_URL}/albums/{album_id}"
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
    url = f"{IMMICH_BASE_URL}/assets/{asset_id}/thumbnail"
    res = requests.get(url, headers=HEADERS)

    if res.status_code != 200:
        logger.warning(f"Thumbnail failed: {asset_id}")
        return None

    return res.content


def update_rating(asset_id: str, rating: int) -> None:
    """
    Update asset rating in Immich.

    :param asset_id: Asset ID
    :param rating: Star rating
    """
    url = f"{IMMICH_BASE_URL}/assets/{asset_id}"
    res = requests.patch(url, headers=HEADERS, json={"rating": rating})

    if res.status_code not in (200, 204):
        logger.error(f"Failed to update {asset_id}: {res.text}")


# =======================
# WORKER
# =======================


def process_asset(
    asset: Dict[str, Any],
    model: tf.keras.Model,
    overwrite: bool
) -> None:
    """
    Process a single asset.

    :param asset: Asset dict
    :param model: NIMA model
    :param overwrite: Overwrite existing ratings
    """
    asset_id: Optional[str] = asset.get("id")
    rating: Optional[int] = asset.get("rating")

    if not asset_id:
        return

    if rating is not None and not overwrite:
        return

    img = download_thumbnail(asset_id)
    if not img:
        return

    try:
        tensor = preprocess(img)
        score = predict_score(model, tensor)
        stars = to_stars(score)

        logger.info(f"{asset_id} -> {score:.2f} ({stars}★)")
        update_rating(asset_id, stars)

    except Exception as e:
        logger.exception(f"Error processing {asset_id}: {e}")


# =======================
# MAIN
# =======================


def run(album_id: str, overwrite: bool) -> None:
    """
    Run full pipeline.

    :param album_id: Album ID
    :param overwrite: Overwrite existing ratings
    """
    logger.info("Starting run")
    logger.info(f"Album: {album_id}")
    logger.info(f"Workers: {MAX_WORKERS}, Batch: {BATCH_SIZE}")

    model = load_model()
    assets = get_assets(album_id)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []

        for i in range(0, len(assets), BATCH_SIZE):
            batch = assets[i:i + BATCH_SIZE]

            for asset in batch:
                futures.append(
                    executor.submit(process_asset, asset, model, overwrite)
                )

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass

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

    args = parser.parse_args()

    run(album_id=args.album_id, overwrite=args.overwrite)