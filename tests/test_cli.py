from project_W_runner.__main__ import main

from click.testing import CliRunner


def test_project_W_runner_cli():
    runner = CliRunner()
    result = runner.invoke(main, ())
    assert result.exit_code == 0
