import click

from typing import Optional
from pathlib import Path
from project_W_runner.config import loadConfig
from project_W_runner.runner import Runner
import asyncio


@click.command()
@click.option("--customConfigPath", type=str, required=False)
def main(customconfigpath: Optional[str] = None):
    config = loadConfig([Path(customconfigpath)]) if customconfigpath else loadConfig()
    runner = Runner(
        backend_url=config["backendURL"],
        token=config["runnerToken"],
        torch_device=config.get("torchDevice"),
        model_cache_dir=config.get("modelCacheDir")
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
