#!/usr/bin/env python3
"""
immich_stack_similar.py
========================

Scan an Immich album, find visually-similar photos using a TensorFlow
(MobileNetV2) embedding model, and group them into Immich "stacks".

Overview
--------
1. Read ``IMMICH_BASE_URL`` and ``API_KEY`` from a ``.env`` file.
2. Resolve the target album (by name or id) and list its image assets.
3. Download a thumbnail for every image and compute a feature embedding
   with a pretrained MobileNetV2 (ImageNet weights, no top layer).
4. Build a similarity graph: two assets are connected if their cosine
   similarity is >= ``--strictness``. Connected components of this graph
   become "stacks", so a stack can contain more than two images (e.g.
   A~B and B~C produces a single 3-image stack even if A and C are not
   directly similar enough).
5. Within each stack, pick the "primary" asset: the image with the
   highest ``exifInfo.rating``, falling back to the highest resolution
   (``exifImageWidth * exifImageHeight``) as a tie-breaker (including
   when no image in the stack has a rating at all).
6. Either print what *would* be stacked (``--dry-run``) or call the
   Immich API to create the stacks, with the chosen primary asset
   listed first.

Usage
-----
.. code-block:: bash

    python immich_stack_similar.py --album "Summer Trip" --strictness 0.9
    python immich_stack_similar.py --album <album-uuid> --dry-run

Requirements
------------
``pip install -r requirements.txt`` (python-dotenv, requests, numpy,
tensorflow, pillow, tqdm).

.. note::
   The Immich REST API has changed shape across releases. This script
   targets the ``POST /api/stacks`` endpoint with a body of
   ``{"assetIds": [...]}``. If your Immich server uses a different
   version/endpoint, check ``<IMMICH_BASE_URL>/api/doc`` (the Swagger
   UI) and adjust :meth:`ImmichClient.create_stack` accordingly.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

# TensorFlow / Keras is a heavy import; keeping it at module scope means
# it only happens once and any import errors surface immediately.
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
)

from scrollingLogDisplay import ScrollingLogDisplay, ScrollingLogHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("immich_stack_similar")


# ---------------------------------------------------------------------------
# Config / environment
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """
    Resolved runtime configuration.

    :ivar base_url: Immich server base URL, e.g. ``https://immich.example.com``.
    :ivar api_key: Immich API key used for authentication.
    """

    base_url: str
    api_key: str


def load_config(env_path: str = ".env") -> Config:
    """
    Load ``IMMICH_BASE_URL`` and ``API_KEY`` from a ``.env`` file.

    :param env_path: Path to the ``.env`` file to load.
    :type env_path: str
    :return: A populated :class:`Config` instance.
    :rtype: Config
    :raises SystemExit: If the ``.env`` file is missing required values.
    """
    if not os.path.isfile(env_path):
        logger.warning("No .env file found at %s; relying on existing environment variables.", env_path)
    load_dotenv(dotenv_path=env_path)

    base_url = os.environ.get("IMMICH_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("API_KEY", "")

    missing = [name for name, val in (("IMMICH_BASE_URL", base_url), ("API_KEY", api_key)) if not val]
    if missing:
        logger.error("Missing required .env values: %s", ", ".join(missing))
        raise SystemExit(1)

    return Config(base_url=base_url, api_key=api_key)


# ---------------------------------------------------------------------------
# Immich API client
# ---------------------------------------------------------------------------
class ImmichClient:
    """
    Thin wrapper around the subset of the Immich REST API needed to list
    album assets, download thumbnails, and create stacks.

    :param base_url: Immich server base URL (no trailing slash).
    :param api_key: Immich API key.
    :param timeout: Per-request timeout in seconds.
    """

    def __init__(self, base_url: str, api_key: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"x-api-key": api_key, "Accept": "application/json"})

    def _url(self, path: str) -> str:
        """Build a full API URL from a path fragment starting with ``/``."""
        return f"{self.base_url}/api{path}"

    def find_album(self, identifier: str) -> Dict:
        """
        Resolve an album by UUID or exact/partial name match.

        :param identifier: Album UUID or album name.
        :type identifier: str
        :return: The album's JSON representation (includes its assets).
        :rtype: dict
        :raises RuntimeError: If no matching album can be found.
        """
        # First, try treating the identifier as an album id directly.
        resp = self.session.get(self._url(f"/albums/{identifier}"), timeout=self.timeout)
        if resp.status_code == 200:
            return resp.json()

        # Fall back to listing all albums and matching by name.
        resp = self.session.get(self._url("/albums"), timeout=self.timeout)
        resp.raise_for_status()
        albums = resp.json()

        matches = [a for a in albums if a.get("albumName") == identifier]
        if not matches:
            # Try a case-insensitive substring match as a last resort.
            lowered = identifier.lower()
            matches = [a for a in albums if lowered in a.get("albumName", "").lower()]

        if not matches:
            raise RuntimeError(f"No album found matching '{identifier}'.")
        if len(matches) > 1:
            names = ", ".join(a["albumName"] for a in matches)
            logger.warning("Multiple albums matched '%s' (%s); using the first.", identifier, names)

        album_id = matches[0]["id"]
        resp = self.session.get(self._url(f"/albums/{album_id}"), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_image_assets(self, album: Dict) -> List[Dict]:
        """
        Extract the image (non-video) assets from an album payload.

        :param album: Album JSON as returned by :meth:`find_album`.
        :type album: dict
        :return: List of asset dicts of type ``IMAGE``.
        :rtype: list[dict]
        """
        assets = album.get("assets", [])
        images = [a for a in assets if a.get("type", "IMAGE") == "IMAGE"]
        logger.info("Album '%s' contains %d assets (%d images).", album.get("albumName"), len(assets), len(images))
        return images

    def get_asset(self, asset_id: str) -> Dict:
        """
        Fetch full asset detail, including ``exifInfo`` (rating, pixel
        dimensions, etc.). The asset objects embedded in an album payload
        are sometimes a slimmer projection, so this is used whenever
        rating/resolution data is required.

        :param asset_id: Immich asset id.
        :type asset_id: str
        :return: Full asset JSON.
        :rtype: dict
        :raises requests.HTTPError: If the asset cannot be fetched.
        """
        resp = self.session.get(self._url(f"/assets/{asset_id}"), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def download_thumbnail(self, asset_id: str, size: str = "preview") -> bytes:
        """
        Download a thumbnail image for the given asset.

        :param asset_id: Immich asset id.
        :type asset_id: str
        :param size: Thumbnail size, either ``"thumbnail"`` or ``"preview"``.
        :type size: str
        :return: Raw image bytes.
        :rtype: bytes
        :raises requests.HTTPError: If the download fails.
        """
        resp = self.session.get(
            self._url(f"/assets/{asset_id}/thumbnail"),
            params={"size": size},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.content

    def create_stack(self, asset_ids: Sequence[str]) -> None:
        """
        Create a stack containing the given assets.

        :param asset_ids: Asset ids to group into a single stack, in the
            desired order. Immich treats the first id as the stack's
            primary asset, so callers should pass the highest-rated /
            highest-resolution asset first (see
            :func:`order_stack_by_rating_and_resolution`).
        :type asset_ids: Sequence[str]
        :raises requests.HTTPError: If the API rejects the request.
        """
        resp = self.session.post(
            self._url("/stacks"),
            json={"assetIds": list(asset_ids)},
            timeout=self.timeout,
        )
        resp.raise_for_status()


# ---------------------------------------------------------------------------
# Similarity model
# ---------------------------------------------------------------------------
class SimilarityModel:
    """
    Wraps a pretrained MobileNetV2 (ImageNet weights) used purely as a
    feature extractor: the classification head is removed and global
    average pooling produces a fixed-length embedding per image.

    :param input_size: Square input resolution expected by the network.
    """

    def __init__(self, input_size: int = 224) -> None:
        self.input_size = input_size
        logger.info("Loading MobileNetV2 feature extractor (this may download weights on first run)...")
        self.model = MobileNetV2(
            input_shape=(input_size, input_size, 3),
            include_top=False,
            weights="imagenet",
            pooling="avg",
        )

    def preprocess(self, raw_bytes: bytes) -> np.ndarray:
        """
        Decode raw image bytes and prepare them for the network.

        :param raw_bytes: Raw (encoded) image bytes, e.g. JPEG/PNG.
        :type raw_bytes: bytes
        :return: A ``(H, W, 3)`` float32 array ready for batching.
        :rtype: numpy.ndarray
        """
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        img = img.resize((self.input_size, self.input_size))
        arr = tf.keras.preprocessing.image.img_to_array(img)
        return preprocess_input(arr)

    def embed_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Compute L2-normalized embeddings for a batch of preprocessed images.

        :param batch: Array of shape ``(N, H, W, 3)``.
        :type batch: numpy.ndarray
        :return: Array of shape ``(N, D)`` with unit-norm rows, so that a
            dot product between rows equals cosine similarity.
        :rtype: numpy.ndarray
        """
        embeddings = self.model.predict(batch, verbose=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms


# ---------------------------------------------------------------------------
# Clustering (Union-Find over a similarity threshold graph)
# ---------------------------------------------------------------------------
class UnionFind:
    """
    Disjoint-set-union structure used to group images into stacks of any
    size (not just pairs) via transitive similarity: if A~B and B~C then
    A, B and C all end up in the same stack even if A and C alone would
    not have passed the similarity threshold.

    :param n: Number of elements to track, indexed ``0..n-1``.
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """
        Find the representative (root) of the set containing ``x``,
        applying path compression.

        :param x: Element index.
        :return: Root index representing ``x``'s set.
        """
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        """
        Merge the sets containing ``a`` and ``b`` (union by rank).

        :param a: First element index.
        :param b: Second element index.
        """
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> Dict[int, List[int]]:
        """
        :return: Mapping of root index -> list of member indices.
        :rtype: dict[int, list[int]]
        """
        result: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            result.setdefault(root, []).append(i)
        return result


def cluster_by_similarity(embeddings: np.ndarray, threshold: float) -> List[List[int]]:
    """
    Group embedding indices into clusters using a cosine-similarity
    threshold graph and connected components (via :class:`UnionFind`).

    :param embeddings: Unit-norm embeddings, shape ``(N, D)``.
    :type embeddings: numpy.ndarray
    :param threshold: Minimum cosine similarity (0-1) required to link
        two images. Higher values require closer matches, producing
        smaller/fewer stacks; lower values are more permissive.
    :type threshold: float
    :return: List of clusters, each a list of embedding indices. Only
        clusters with 2 or more members represent an actual stack.
    :rtype: list[list[int]]
    """
    n = embeddings.shape[0]
    uf = UnionFind(n)

    # Cosine similarity between all pairs (embeddings are already unit norm).
    sim_matrix = embeddings @ embeddings.T

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                uf.union(i, j)

    clusters = [members for members in uf.groups().values() if len(members) > 1]
    return clusters


def order_stack_by_rating_and_resolution(client: ImmichClient, members: List[Dict]) -> List[Dict]:
    """
    Reorder a cluster's assets so the "best" one is first.

    Immich treats the first id in a stack's asset list as the stack's
    primary asset, so this picks the primary using, in priority order:

    1. Highest ``exifInfo.rating`` (assets with no rating are treated as
       the lowest possible rating, so any rated image outranks an
       unrated one).
    2. Highest resolution (``exifImageWidth * exifImageHeight``) as a
       tie-breaker when ratings are equal (including when none of the
       images in the stack are rated).

    :param client: Configured :class:`ImmichClient`, used to fetch full
        asset detail (``exifInfo`` is not always present on the slimmer
        asset objects embedded in an album listing).
    :type client: ImmichClient
    :param members: Asset dicts belonging to one cluster/stack.
    :type members: list[dict]
    :return: The same assets, sorted best-first (highest rating, then
        highest resolution).
    :rtype: list[dict]
    """
    scored: List[Tuple[int, int, Dict]] = []

    for asset in members:
        try:
            detail = client.get_asset(asset["id"])
        except Exception as exc:  # noqa: BLE001 - fall back to unscored placement rather than failing the stack
            logger.warning("Could not fetch exif detail for %s: %s", asset.get("originalFileName", asset["id"]), exc)
            detail = asset

        exif = detail.get("exifInfo") or {}
        rating = exif.get("rating")
        rating_score = rating if isinstance(rating, (int, float)) else -1

        width = exif.get("exifImageWidth") or 0
        height = exif.get("exifImageHeight") or 0
        resolution_score = width * height

        scored.append((rating_score, resolution_score, asset))
        logger.debug(
            "  candidate %s: rating=%s resolution=%dx%d",
            asset.get("originalFileName", asset["id"]),
            rating if rating is not None else "none",
            width,
            height,
        )

    # Sort descending by (rating, resolution); Python's sort is stable so
    # ties beyond that keep their original (cluster) order.
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    best_rating, best_resolution, best_asset = scored[0]
    logger.info(
        "  primary: %s (rating=%s, resolution=%d px)",
        best_asset.get("originalFileName", best_asset["id"]),
        best_rating if best_rating != -1 else "none",
        best_resolution,
    )

    return [item[2] for item in scored]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def fetch_embeddings(
    client: ImmichClient,
    model: SimilarityModel,
    assets: List[Dict],
    display: ScrollingLogDisplay,
    workers: int = 8,
) -> Tuple[List[Dict], np.ndarray]:
    """
    Download thumbnails concurrently and compute embeddings for every
    asset, reporting progress via ``display``.

    Downloads happen in a thread pool (I/O bound); the actual TensorFlow
    inference is run once as a single batch on the main thread, since
    Keras models are not designed for concurrent calls across threads.

    :param client: Configured :class:`ImmichClient`.
    :param model: Configured :class:`SimilarityModel`.
    :param assets: Asset dicts to process.
    :param display: Progress/log display to update as work completes.
    :param workers: Number of concurrent download threads.
    :return: Tuple of (assets actually embedded, their embeddings array).
        Assets that failed to download are skipped and logged.
    :rtype: tuple[list[dict], numpy.ndarray]
    """
    preprocessed: Dict[int, np.ndarray] = {}

    def _download_and_preprocess(idx_asset: Tuple[int, Dict]) -> Tuple[int, Optional[np.ndarray]]:
        idx, asset = idx_asset
        try:
            raw = client.download_thumbnail(asset["id"])
            arr = model.preprocess(raw)
            return idx, arr
        except Exception as exc:  # noqa: BLE001 - log and continue, don't kill the whole run
            display.log(f"WARN  failed to download/preprocess {asset.get('originalFileName', asset['id'])}: {exc}")
            return idx, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_and_preprocess, item) for item in enumerate(assets)]
        for future in as_completed(futures):
            idx, arr = future.result()
            if arr is not None:
                preprocessed[idx] = arr
                display.log(f"OK    embedded {assets[idx].get('originalFileName', assets[idx]['id'])}")
            display.advance(1)

    if not preprocessed:
        return [], np.empty((0, 0))

    ordered_indices = sorted(preprocessed.keys())
    batch = np.stack([preprocessed[i] for i in ordered_indices], axis=0)
    embeddings = model.embed_batch(batch)
    kept_assets = [assets[i] for i in ordered_indices]
    return kept_assets, embeddings


def run(config: Config, album_identifier: str, strictness: float, dry_run: bool, workers: int) -> None:
    """
    End-to-end pipeline: fetch album, embed images, cluster, and stack.

    :param config: Loaded :class:`Config` with Immich credentials.
    :param album_identifier: Album name or UUID to scan.
    :param strictness: Cosine-similarity threshold (0-1) for grouping.
    :param dry_run: If ``True``, only report what would be stacked.
    :param workers: Number of concurrent thumbnail-download threads.
    """
    client = ImmichClient(config.base_url, config.api_key)
    album = client.find_album(album_identifier)
    assets = client.get_image_assets(album)

    if len(assets) < 2:
        logger.info("Fewer than 2 images in album; nothing to compare.")
        return

    model = SimilarityModel()

    display = ScrollingLogDisplay(total=len(assets), desc="Embedding images")
    handler = ScrollingLogHandler(display)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        kept_assets, embeddings = fetch_embeddings(client, model, assets, display, workers=workers)
    finally:
        display.close()
        root_logger.removeHandler(handler)

    if len(kept_assets) < 2:
        logger.warning("Fewer than 2 images were successfully embedded; nothing to compare.")
        return

    logger.info("Clustering %d images at strictness=%.3f ...", len(kept_assets), strictness)
    clusters = cluster_by_similarity(embeddings, threshold=strictness)

    if not clusters:
        logger.info("No similar-image groups found above the strictness threshold.")
        return

    logger.info("Found %d candidate stack(s).", len(clusters))
    for cluster in clusters:
        members = [kept_assets[i] for i in cluster]
        names = [a.get("originalFileName", a["id"]) for a in members]
        logger.info("Ranking primary asset for stack: %s", ", ".join(names))

        ordered_members = order_stack_by_rating_and_resolution(client, members)
        ordered_ids = [a["id"] for a in ordered_members]
        ordered_names = [a.get("originalFileName", a["id"]) for a in ordered_members]

        if dry_run:
            logger.info(
                "[DRY RUN] Would stack %d images (primary first): %s",
                len(ordered_ids),
                ", ".join(ordered_names),
            )
            continue

        try:
            client.create_stack(ordered_ids)
            logger.info("Stacked %d images (primary first): %s", len(ordered_ids), ", ".join(ordered_names))
        except Exception as exc:  # noqa: BLE001 - keep going with remaining stacks
            logger.error("Failed to create stack for %s: %s", ", ".join(ordered_names), exc)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    :param argv: Argument list to parse; defaults to ``sys.argv[1:]``.
    :type argv: Sequence[str] or None
    :return: Parsed argument namespace with ``album``, ``strictness``,
        ``dry_run``, ``workers`` and ``env_file`` attributes.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Find and stack visually similar images in an Immich album using a TensorFlow similarity model.",
    )
    parser.add_argument(
        "--album",
        required=True,
        help="Immich album name or UUID to scan.",
    )
    parser.add_argument(
        "--strictness",
        type=float,
        default=0.9,
        help=(
            "Cosine-similarity threshold in [0, 1] required to group two images together. "
            "Higher = stricter/fewer matches, lower = looser/more matches. Default: 0.9."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the stacks that would be created without modifying the Immich album.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent threads used to download thumbnails. Default: 8.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to the .env file containing IMMICH_BASE_URL and API_KEY. Default: .env",
    )
    args = parser.parse_args(argv)

    if not 0.0 <= args.strictness <= 1.0:
        parser.error("--strictness must be between 0.0 and 1.0")

    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Script entry point.

    :param argv: Optional argument list (used for testing); defaults to
        ``sys.argv[1:]`` when ``None``.
    :type argv: Sequence[str] or None
    :return: Process exit code (``0`` on success).
    :rtype: int
    """
    args = parse_args(argv)
    config = load_config(args.env_file)

    try:
        run(
            config=config,
            album_identifier=args.album,
            strictness=args.strictness,
            dry_run=args.dry_run,
            workers=args.workers,
        )
    except Exception as exc:  # noqa: BLE001 - top-level guard for a clean CLI error message
        logger.error("Fatal error: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())