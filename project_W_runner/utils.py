from typing import Callable, Optional

import tqdm
from whisper import load_model


def transcribe(
    audio_file: str,
    model: Optional[str],
    language: Optional[str],
    progress_callback: Callable[[float], None],
    device: Optional[str],
    model_cache_dir: Optional[str],
) -> dict[str, str]:
    """
    Transcribe the given audio using the given model and language. Returns the dictionary of all the
    information returned by the Whisper invocation. The progress_callback is called periodically with
    the progress of the transcription, as a float between 0 and 1. If device is not None, it will be
    used as the device for the model.
    """

    # Heavy inspiration from <https://github.com/ssciwr/vink/blob/main/vink.py>
    def monkeypatching_tqdm(progress_cb):
        def _monkeypatching_tqdm(
            total=None,
            ncols=None,
            unit=None,
            unit_scale=True,
            unit_divisor=None,
            disable=False,
        ):
            class TqdmMonkeypatchContext:
                def __init__(self) -> None:
                    self.progress = 0.0

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def update(self, value):
                    if unit_divisor:
                        value = value / unit_divisor
                    self.progress += value
                    progress_cb(self.progress / total)

            if unit_divisor:
                total = total / unit_divisor

            return TqdmMonkeypatchContext()

        return _monkeypatching_tqdm

    # TODO: Load the model for the correct language.
    model = load_model(model or "base", device=device, download_root=model_cache_dir)
    tqdm.tqdm = monkeypatching_tqdm(progress_callback)

    progress_callback(0.0)
    ret = model.transcribe(audio_file)
    # Just in case the progress_callback was not called with 1.0, do that now.
    progress_callback(1.0)

    return ret
