import asyncio
import os
from pathlib import Path

import click

from ._version import __version__
from .config import load_config
from .logger import get_logger

logger = get_logger("project-W-runner")


@click.command()
@click.version_option(__version__)
@click.option(
    "--custom_config_path",
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        allow_dash=False,
        path_type=Path,
    ),
    required=False,
    help="Path to search for the config.yml file in addition to the users and sites config paths (xdg dirs on Linux) and the current working directory.",
)
def main(custom_config_path: Path | None):
    logger.info(f"Running application version {__version__}")

    config = load_config([custom_config_path]) if custom_config_path else load_config()

    # set env vars first before importing Runner and prefetch code because that would trigger the code that reads the env var
    os.environ["PYANNOTE_CACHE"] = str(config.whisper_settings.model_cache_dir)
    os.environ["HF_HOME"] = str(config.whisper_settings.model_cache_dir)

    logger.info("Trying to import runner and WhisperX code now...")
    from .runner import Runner
    from .utils import prefetch_all_models

    logger.info("Import successful")

    if config.skip_model_prefetch:
        logger.warning(
            "Skipping model prefetching, this might lead to failing jobs due to not being able to fetch models and significantly higher job processing times!"
        )
    else:
        prefetch_all_models(config.whisper_settings)

    runner = Runner(
        config=config,
        backend_url=config.backend_url,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
