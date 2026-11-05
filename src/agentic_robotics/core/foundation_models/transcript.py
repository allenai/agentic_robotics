"""Define a class that logs a transcript of an ongoing conversation with a foundation model."""

import base64
import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from openai.types.responses import ResponseInputItemParam

from agentic_robotics.core.foundation_models.message_conversions import create_openai_message
from agentic_robotics.core.foundation_models.model_client import History

ImageReplacer = Callable[[bytes, str], str]
"""Given an image's decoded bytes and MIME type, output text to replace its data URL."""

MIME_TYPE_TO_FILE_SUFFIX = {"image/jpeg": ".jpg", "image/png": ".png"}

_RESET_TEXT = "----------[EPISODE HAS BEEN RESET]----------"
"""Text used to populate the transcript-only message added to the transcript upon episode reset."""


def replace_images(item: ResponseInputItemParam, replace: ImageReplacer) -> ResponseInputItemParam:
    """Return a copy of a conversation item with every inline image URL replaced.

    :param item: Conversation item whose images are replaced in the output copy
    :param replace: Function producing the text written in place of each image URL
    :return: Copy of the input item (original input is unmodified)
    """
    content_key = "content" if "content" in item else "output" if "output" in item else None
    if content_key is None or isinstance(item.get(content_key), str):
        return item

    item_copy = copy.copy(item)
    item_copy[content_key] = [_replace_images_in_block(b, replace) for b in item[content_key]]
    return item_copy


def _replace_images_in_block(block: dict, replace: ImageReplacer) -> dict:
    """Replace the image URL in a content block, if it contains an image."""
    if block["type"] != "input_image":
        return block

    image_url = str(block["image_url"])
    prefix, separator, b64_data = image_url.partition(";base64,")
    if not separator:
        return block  # Hosted URL instead of an inline image; no need to replace

    mime_type = prefix.removeprefix("data:")
    image_bytes = base64.b64decode(b64_data)

    redacted_block = copy.copy(block)
    redacted_block["image_url"] = replace(image_bytes, mime_type)

    return redacted_block


class Transcript:
    """A conversation history that logs each item to file as the conversation continues."""

    def __init__(self, out_dir: Path) -> None:
        """Initialize an empty conversation history.

        :param out_dir: Path to the output directory to receive the transcript and its images
        """
        self._items: History = []

        self._out_dir = out_dir
        self._image_dir = out_dir / "images"
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = out_dir / "transcript.jsonl"

        self._num_images: int = 0
        self._episode_idx: int = 0
        self._episode_start_idx: int = 0
        """Index tracking the first item in the transcript's history for the current episode."""

    @property
    def current_episode(self) -> History:
        """Retrieve the conversation history for the current episode."""
        return self._items[self._episode_start_idx :]

    def append(self, item: ResponseInputItemParam, metadata: dict[str, Any] | None = None) -> None:
        """Add one item to the conversation, logging it and any images it includes.

        :param item: Item in the conversation, compatible with the OpenAI Responses API
        :param metadata: Optional additional metadata logged to the transcript output file
        """
        self._items.append(item)

        metadata = {**(metadata or {}), "episode_idx": self._episode_idx}

        loggable_item = replace_images(item, self._save_image)
        loggable_item = {**loggable_item, "metadata": metadata}

        with self._log_path.open(mode="a") as log_file:
            log_file.write(json.dumps(loggable_item) + "\n")

    def extend(self, items: History) -> None:
        """Add several items to the conversation, in the given order."""
        for item in items:
            self.append(item)

    def start_new_episode(self, initial_message: ResponseInputItemParam) -> None:
        """Begin a new episode (i.e., conversation used as model context) in the transcript.

        :param initial_message: Prompt from the user used to initialize the new conversation
        """
        reset_message = create_openai_message(text=_RESET_TEXT, role="system")
        self.append(reset_message)

        self._episode_idx += 1
        self.append(initial_message)
        self._episode_start_idx = len(self._items) - 1

    def _save_image(self, data: bytes, mime_type: str) -> str:
        """Write an image alongside the transcript, returning the resulting filename."""
        suffix = MIME_TYPE_TO_FILE_SUFFIX.get(mime_type, ".bin")
        image_path = self._image_dir / f"image_{self._num_images:04d}{suffix}"
        image_path.write_bytes(data)
        self._num_images += 1

        return image_path.name
