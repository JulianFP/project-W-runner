{
  lib,
  ffmpeg,
  python3Packages,
}:

python3Packages.buildPythonPackage rec {
  pname = "project_W_runner";
  version = "0.0.1";
  format = "setuptools";

  src = ../../.;

  nativeBuildInputs = with python3Packages; [
    setuptools-scm
  ];

  propagatedBuildInputs = with python3Packages; [
    ffmpeg
    aiohttp
    click
    jsonschema
    openai-whisper
    platformdirs
    pyaml-env
  ];

  nativeCheckInputs = with python3Packages; [
    pytestCheckHook
    pytest-cov
  ];
  pythonImportsCheck = [ pname ];

  #hardcode version so that setuptools-scm works without .git folder:
  SETUPTOOLS_SCM_PRETEND_VERSION = version;

  meta = {
    description = "Runner for Project-W";
    homepage = "https://github.com/JulianFP/project-W-runner";
    license = lib.licenses.agpl3Only;
    mainProgram = pname;
  };
}
