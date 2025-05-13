import asyncio
import os
from pathlib import Path
from typing import Optional

import click

from .config import load_config


@click.command()
@click.option("--custom_config_path", type=str, required=False)
def main(custom_config_path: Optional[str] = None):
    config = load_config([Path(custom_config_path)]) if custom_config_path else load_config()

    # set env vars first before importing Runner and prefetch code because that would trigger the code that reads the env var
    os.environ["PYANNOTE_CACHE"] = str(config.whisper_settings.model_cache_dir)
    os.environ["HF_HOME"] = str(config.whisper_settings.model_cache_dir)
    from .runner import Runner
    from .utils import prefetch_all_models

    if not config.skip_model_prefetch:
        prefetch_all_models(config.whisper_settings)
    runner = Runner(
        config=config,
        backend_url=config.backend_url,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
