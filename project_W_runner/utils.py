import gc
from io import StringIO
from typing import Callable

import numpy as np
import torch
import whisperx
from whisperx import alignment, asr, diarize, utils

from .logger import get_logger
from .models.request_data import (
    AlignmentProcessingSettings,
    JobModelEnum,
    supported_alignment_languages,
)
from .models.response_data import JobSettings
from .models.settings import WhisperSettings

logger = get_logger("project-W-runner")


def model_cleanup(model):
    gc.collect()
    torch.cuda.empty_cache()
    del model


def prefetch_all_models(whisper_settings: WhisperSettings):
    """
    Should be called at runner startup before runner registers to backend
    """
    # prefetch diarization model (and check access to it)
    # do this first because this is most likely to fail (because of the required hf token), and the other downloads take a while
    logger.info("Starting prefetching of diarization models now. Please wait...")
    diarize_model = diarize.DiarizationPipeline(
        device=whisper_settings.torch_device,
        use_auth_token=whisper_settings.hf_token.get_secret_value(),
    )
    model_cleanup(diarize_model)
    logger.info("All diarization models fetched successfully")

    # prefetch all whisper models
    logger.info("Starting prefetching of whisper models now. Please wait...")
    for model in JobModelEnum:
        loaded_model = whisperx.load_model(
            model,
            whisper_settings.torch_device,
            download_root=str(whisper_settings.model_cache_dir),
            compute_type=whisper_settings.compute_type,
        )
        model_cleanup(loaded_model)
    logger.info("All whisper models fetched successfully")

    # prefetch all alignment models
    logger.info("Starting prefetching of alignment models now. Please wait...")
    for language in supported_alignment_languages:
        loaded_model = whisperx.load_align_model(
            language_code=language,
            device=whisper_settings.torch_device,
            model_dir=str(whisper_settings.model_cache_dir),
        )
        model_cleanup(loaded_model)
    logger.info("All alignment models fetched successfully")


def transcribe(
    audio_file: str,
    job_settings: JobSettings,
    whisper_settings: WhisperSettings,
    progress_callback: Callable[[float], None],
) -> dict[str, StringIO]:
    """
    Transcribe the given audio using the given model and language. Returns the dictionary of all the
    information returned by the Whisper invocation. The progress_callback is called periodically with
    the progress of the transcription, as a float between 0 and 1. If device is not None, it will be
    used as the device for the model.
    """

    # overwrite the print method inside the asr and alignment modules of whisperx which contain progress printing and detected language
    def intercept_stdout(out: str):
        if out.startswith("Progress: ") and out.endswith("%..."):
            progress = out.removeprefix("Progress: ").removesuffix("%...").strip()
            progress_callback(float(progress))
        elif out.startswith("Detected language: ") and out.endswith(" in first 30s of audio..."):
            detected_language = out.removeprefix("Detected language: ").split(" ")[0].strip()
            if job_settings.language is None and job_settings.alignment is not None:
                if detected_language is None:
                    raise Exception(
                        "Automatic language detection failed: alignment not possible without knowing the language of the transcription. Either select a language explicitly or disable alignment!"
                    )
                if detected_language not in supported_alignment_languages:
                    raise Exception(
                        f"The detected language '{detected_language}' is not supported for alignment. Either select a different language explicitly or disable alignment!"
                    )
        print(out)

    setattr(asr, "print", intercept_stdout)
    setattr(alignment, "print", intercept_stdout)

    # overwrite the ResultWriter class in the utils module to be able to use StringIO instead of actual files for output
    in_memory_files = {}

    def new_result_writer_call(self, result: dict, audio_path: str, options: dict):
        self.in_memory_file = StringIO()
        in_memory_files[self.extension] = self.in_memory_file
        self.write_result(result, file=self.in_memory_file, options=options)

    utils.ResultWriter.__call__ = new_result_writer_call

    # preperations
    if (increment := job_settings.asr_settings.temperature_increment_on_fallback) is not None:
        temperatures = tuple(
            np.arange(job_settings.asr_settings.temperature, 1.0 + 1e-6, increment)
        )
    else:
        temperatures = [job_settings.asr_settings.temperature]
    vad_options = job_settings.vad_settings.model_dump()
    vad_options.pop("chunk_size")
    whisperx_model = whisperx.load_model(
        job_settings.model,
        whisper_settings.torch_device,
        download_root=str(whisper_settings.model_cache_dir),
        local_files_only=True,
        compute_type=whisper_settings.compute_type,
        task=job_settings.task,
        language=job_settings.language,
        asr_options={
            "beam_size": job_settings.asr_settings.beam_size,
            "patience": job_settings.asr_settings.patience,
            "length_penalty": job_settings.asr_settings.length_penalty,
            "temperatures": temperatures,
            "compression_ratio_threshold": job_settings.asr_settings.compression_ratio_threshold,
            "log_prob_threshold": job_settings.asr_settings.log_prob_threshold,
            "no_speech_threshold": job_settings.asr_settings.no_speech_threshold,
            "condition_on_previous_text": False,
            "initial_prompt": job_settings.asr_settings.initial_prompt,
            "suppress_tokens": job_settings.asr_settings.suppress_tokens,
            "suppress_numerals": job_settings.asr_settings.suppress_numerals,
        },
        vad_options=vad_options,
    )
    audio = whisperx.load_audio(audio_file)

    progress_callback(0.0)

    # transcription
    result = whisperx_model.transcribe(
        audio,
        batch_size=whisper_settings.batch_size,
        chunk_size=job_settings.vad_settings.chunk_size,
        print_progress=True,
        combined_progress=True,
    )
    model_cleanup(whisperx_model)

    # new_detect_language should already handle problems with language detection before transcription. This just ensures that the language is definitely not None
    used_language = job_settings.language or result.get("language") or "en"

    # alignment
    if job_settings.alignment is not None:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=used_language,
            device=whisper_settings.torch_device,
            model_dir=str(whisper_settings.model_cache_dir),
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            whisper_settings.torch_device,
            return_char_alignments=job_settings.alignment.return_char_alignments,
            interpolate_method=job_settings.alignment.interpolate_method,
            print_progress=True,
            combined_progress=True,
        )
        model_cleanup(align_model)

    # diarization
    if job_settings.diarization is not None:
        diarize_model = diarize.DiarizationPipeline(
            device=whisper_settings.torch_device,
            use_auth_token=whisper_settings.hf_token.get_secret_value(),
        )
        diarize_segments = diarize_model(
            audio,
            min_speakers=job_settings.diarization.min_speakers,
            max_speakers=job_settings.diarization.max_speakers,
        )
        result = whisperx.assign_word_speakers(diarize_segments, result)
        model_cleanup(diarize_model)

    # output writing
    result = dict(result)
    result["language"] = used_language
    writer = utils.get_writer("all", ".")
    # this function is actually supposed to take a str instead of a TextIO. The error is due to an incorrect type annotation in whisperx upstream, see https://github.com/m-bain/whisperX/pull/1144 for the pending fix
    options = (
        job_settings.alignment.processing.model_dump()
        if job_settings.alignment is not None
        else AlignmentProcessingSettings().model_dump()
    )
    writer(result, "file", options)

    # Just in case the progress_callback was not called with 1.0, do that now.
    progress_callback(1.0)

    return in_memory_files
