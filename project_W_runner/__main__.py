import click

from project_W_runner.config import loadConfig
from project_W_runner.runner import Runner
import asyncio

@click.command()
@click.option("--customConfigPath", type=str, required=False)
def main(customconfigpath: str = None):
    config = loadConfig([customconfigpath]) if customconfigpath else loadConfig()
    runner = Runner(backend_url=config["backendURL"], token=config["runnerToken"], torch_device=config.get("torchDevice"))
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
