"""Utility definitions for creating files and directories."""

import logging
import time
from pathlib import Path

import imageio as iio
import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def create_unique_dir(parent_dir: Path, dir_prefix: str = "") -> Path:
    """Create a unique directory under the given parent directory.

    :param parent_dir: Parent directory of the created directory
    :param dir_prefix: String prefix of the created directory's name
    :return: Path to the created unique directory
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    unique_dir = parent_dir / f"{dir_prefix}{timestamp}"
    unique_dir.mkdir(parents=True, exist_ok=False)
    return unique_dir


def export_video(
    frames: list[np.ndarray], video_path: Path, *, fps: int = 25, quality: int = 9
) -> bool:
    """Export the given frames as a video to the given path.

    :param frames: Image frames converted into a video
    :param video_path: Output filepath for the exported video
    :param fps: Frames per second of the exported video (defaults to 25)
    :param quality: Video output quality (between 0 and 10; defaults to 9)
    :return: True if the video was successfully exported, else False
    """
    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(video_path, fps=fps, quality=quality)

    for image in frames:
        writer.append_data(image)

    writer.close()

    success = video_path.exists()
    if success:
        logger.info(f"Saved {len(frames)}-frame video to {video_path}")
    return success


def export_image(image: np.ndarray, output_path: Path) -> bool:
    """Export the given image to the specified filepath."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(image).save(output_path)
    return output_path.exists()


class IncrementalVideoWriter:
    """Write video frames to file as they arrive, rather than buffering them until export."""

    def __init__(self, video_path: Path, *, fps: int = 25, quality: int = 9) -> None:
        """Initialize the incremental video writer using a target filepath.

        :param video_path: Output filepath for the exported video
        :param fps: Frames per second of the exported video (defaults to 25)
        :param quality: Video output quality (between 0 and 10; defaults to 9)
        """
        self.video_path = video_path
        self.frames_written = 0

        self._writer = iio.get_writer(
            video_path,
            fps=fps,
            quality=quality,
            output_params=["-movflags", "+frag_keyframe+empty_moov"],
        )
        self._closed = False

    def append(self, frame: np.ndarray) -> None:
        """Encode an additional frame into the open file."""
        if not self._closed:
            self._writer.append_data(frame)
            self.frames_written += 1

    def close(self) -> None:
        """Finalize the file. Safe to call more than once."""
        if not self._closed:
            self._writer.close()
            self._closed = True
            logger.info(f"Saved {self.frames_written}-frame video to {self.video_path}")
