# This is mostly just copied from the backend config code, but with a different schema.

from pathlib import Path
from typing import Dict, List

from jsonschema import Draft202012Validator, ValidationError, validators
from platformdirs import site_config_path, user_cache_dir, user_config_path
from pyaml_env import parse_config

from .logger import get_logger

programName = "project-W-runner"
logger = get_logger(programName)


def extend_with_default(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(property, subschema["default"])

        for error in validate_properties(
            validator,
            properties,
            instance,
            schema,
        ):
            yield error

    return validators.extend(
        validator_class,
        {"properties": set_defaults},
    )


DefaultValidatingValidator = extend_with_default(Draft202012Validator)


class findConfigFileException(Exception):
    pass


class prettyValidationError(ValidationError):
    pass


# schema for config variables. Will be used for both the config file and env vars
schema = {
    "type": "object",
    "properties": {
        "backendURL": {
            "type": "string",
            "pattern": r"^(http|https):\/\/(([a-zA-Z0-9\-]+\.)+[a-zA-Z0-9\-]+|localhost)(:[0-9]+)?((\/[a-zA-Z0-9\-]+)+)?$",
        },
        "runnerToken": {
            "type": "string",
            "pattern": r"^[a-zA-Z0-9_-]+$",
        },
        "torchDevice": {"type": "string"},
        "modelCacheDir": {
            "type": "string",
        },
        "jobTmpDir": {
            "type": "string",
            "default": str(user_cache_dir(appname=programName)),
        },
        "disableOptionValidation": {"type": "boolean", "default": False},
    },
    "required": ["backendURL", "runnerToken"],
    "additionalProperties": False,
}


def findConfigFile(additionalPaths: List[Path] = []) -> Path:
    defaultSearchDirs = [
        user_config_path(appname=programName),
        site_config_path(appname=programName),
        Path(__file__).parent,
        Path.cwd(),
    ]
    searchDirs = additionalPaths + defaultSearchDirs

    for dir in searchDirs:
        configDir = dir / "config.yml"
        if configDir.is_file():
            logger.info("Trying to load config from: " + str(configDir))
            return configDir
    raise findConfigFileException(
        "couldn't find a config.yml file in any search directory. Please add one"
    )


def loadConfig(additionalPaths: List[Path] = []) -> Dict:
    configPath = findConfigFile(additionalPaths)
    config = parse_config(configPath)

    # print warning about option if it is set
    if config.get("disableOptionValidation"):
        logger.warning(
            "'disableOptionValidation' has been enabled in your config. Only do this for development or testing purposes, never in production!"
        )

    # catch exception for validation with jsonscheme to implement disableOptionValidation
    # and for better exception messages in terminal and log messages
    try:
        DefaultValidatingValidator(schema).validate(config)
    except ValidationError as exc:
        if not config.get("disableOptionValidation"):
            msg = ""
            if exc.validator == "required":
                msg = (
                    "A required option is missing from your config.yml file:\n"
                    + exc.message
                    + "\nPlease make sure to define this option. Maybe you made a typo?"
                )
            elif exc.validator == "additionalProperties":
                msg = (
                    "An undefined option has been found in your config.yml file:\n"
                    + exc.message
                    + "\nPlease remove this option from your config. Maybe you made a typo?"
                )
            else:
                msg = (
                    "The option '"
                    + exc.json_path
                    + "' in your config.yml file has has an invalid value:\n"
                    + exc.message
                    + "\nPlease adjust this value. Maybe you made a typo?"
                )
            logger.exception(msg)
            raise prettyValidationError(msg)
        else:
            logger.warning(
                "Your config is invalid, some parts of this program will not work properly! Set 'disableOptionValidation' to false to learn more"
            )

    logger.info("successfully loaded config from: " + str(configPath))
    return config
