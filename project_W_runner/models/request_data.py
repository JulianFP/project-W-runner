from enum import Enum
from typing import Self

from pydantic import BaseModel, Field, model_validator


class JobModelEnum(str, Enum):
    TINY = "tiny"
    TINY_EN = "tiny.en"
    BASE = "base"
    BASE_EN = "base.en"
    SMALL = "small"
    SMALL_EN = "small.en"
    MEDIUM = "medium"
    MEDIUM_EN = "medium.en"
    TURBO = "turbo"
    LARGE = "large"


class JobLangEnum(str, Enum):
    # only support ISO 639 language codes since the full names have duplicates (e.g. Catalan and Valencian or Spanish and Castilian or Dutch and Flemish) and are longer
    AFRIKAANS = "af"
    AMHARIC = "am"
    ARABIC = "ar"
    ASSAMESE = "as"
    AZERBAIJANI = "az"
    BASHKIR = "ba"
    BELARUSIAN = "be"
    BULGARIAN = "bg"
    BENGALI = "bn"
    TIBETAN = "bo"
    BRETON = "br"
    BOSNIAN = "bs"
    CATALAN = "ca"
    CZECH = "cs"
    WELSH = "cy"
    DANISH = "da"
    GERMAN = "de"
    GREEK_MODERN = "el"
    ENGLISH = "en"
    SPANISH = "es"
    ESTONIAN = "et"
    BASQUE = "eu"
    PERSIAN = "fa"
    FINNISH = "fi"
    FAROESE = "fo"
    FRENCH = "fr"
    GALICIAN = "gl"
    GUJARATI = "gu"
    HAUSA = "ha"
    HAWAIIAN = "haw"
    HEBREW = "he"
    HINDI = "hi"
    CROATIAN = "hr"
    HAITIAN = "ht"
    HUNGARIAN = "hu"
    ARMENIAN = "hy"
    INDONESIAN = "id"
    ICELANDIC = "is"
    ITALIAN = "it"
    JAPANESE = "ja"
    JAVANESE = "jw"
    GEORGIAN = "ka"
    KAZAKH = "kk"
    CENTRAL_KHMER = "km"
    KANNADA = "kn"
    KOREAN = "ko"
    LATIN = "la"
    LUXEMBOURGISH = "lb"
    LINGALA = "ln"
    LAO = "lo"
    LITHUANIAN = "lt"
    LATVIAN = "lv"
    MALAGASY = "mg"
    MAORI = "mi"
    MACEDONIAN = "mk"
    MALAYALAM = "ml"
    MONGOLIAN = "mn"
    MARATHI = "mr"
    MALAY = "ms"
    MALTESE = "mt"
    BURMESE = "my"
    NEPALI = "ne"
    DUTCH = "nl"
    NORWEGIAN_NYNORSK = "nn"
    NORWEGIAN = "no"
    OCCITAN = "oc"
    PUNJABI = "pa"
    POLISH = "pl"
    PASHTO = "ps"
    PORTUGUESE = "pt"
    ROMANIAN = "ro"
    RUSSIAN = "ru"
    SANSKRIT = "sa"
    SINDHI = "sd"
    SINHALA = "si"
    SLOVAK = "sk"
    SLOVENIAN = "sl"
    SHONA = "sn"
    SOMALI = "so"
    ALBANIAN = "sq"
    SERBIAN = "sr"
    SUNDANESE = "su"
    SWEDISH = "sv"
    SWAHILI = "sw"
    TAMIL = "ta"
    TELUGU = "te"
    TAJIK = "tg"
    THAI = "th"
    TURKMEN = "tk"
    TAGALOG = "tl"
    TURKISH = "tr"
    TATAR = "tt"
    UKRAINIAN = "uk"
    URDU = "ur"
    UZBEK = "uz"
    VIETNAMESE = "vi"
    YIDDISH = "yi"
    Yoruba = "yo"
    CANTONESE = "yue"
    MANDARIN = "zh"


class DiarizationSettings(BaseModel):
    min_speakers: int | None = Field(
        default=None,
        ge=0,
    )
    max_speakers: int | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def max_must_be_larger_equal_than_min(self) -> Self:
        if (
            self.min_speakers is not None
            and self.max_speakers is not None
            and self.max_speakers < self.min_speakers
        ):
            raise ValueError("max_speakers can't be smaller than min_speakers")
        return self


class AlignmentProcessingSettings(BaseModel):
    highlight_words: bool = False
    max_line_count: int | None = Field(
        default=None,
        ge=1,
    )
    max_line_width: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def max_line_count_needs_max_line_width(self) -> Self:
        if self.max_line_count is not None and self.max_line_width is None:
            raise ValueError("max_line_width can't be None if max_line_count is set")
        return self


class InterpolateMethodEnum(str, Enum):
    NEAREST = "nearest"
    LINEAR = "linear"
    IGNORE = "ignore"


class AlignmentSettings(BaseModel):
    processing: AlignmentProcessingSettings = AlignmentProcessingSettings()
    return_char_alignments: bool = False
    interpolate_method: InterpolateMethodEnum = InterpolateMethodEnum.NEAREST


class TaskEnum(str, Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"


class VadSettings(BaseModel):
    vad_onset: float = Field(
        ge=0.0,
        le=1.0,
        default=0.5,
    )
    vad_offset: float = Field(
        ge=0.0,
        le=1.0,
        default=0.363,
    )
    chunk_size: int = Field(ge=1, le=30, default=30)


class AsrSettings(BaseModel):
    beam_size: int = Field(
        ge=1,
        default=5,
    )
    patience: float = Field(
        gt=0.0,
        default=1.0,
    )
    length_penalty: float = Field(
        ge=0.0,
        le=1.0,
        default=1.0,
    )
    temperature: float = Field(
        ge=0.0,
        default=0.0,
    )
    temperature_increment_on_fallback: float = Field(
        ge=0.0,
        default=0.2,
    )
    compression_ratio_threshold: float = Field(
        ge=0.0,
        default=2.4,
    )
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    initial_prompt: str | None = Field(
        default=None,
        max_length=2000,
    )
    suppress_tokens: list[int] = [-1]
    suppress_numerals: bool = False


supported_alignment_languages = [
    "en",
    "fr",
    "de",
    "es",
    "it",
    "ja",
    "zh",
    "nl",
    "uk",
    "pt",
    "ar",
    "cs",
    "ru",
    "pl",
    "hu",
    "fi",
    "fa",
    "el",
    "tr",
    "da",
    "he",
    "vi",
    "ko",
    "ur",
    "te",
    "hi",
    "ca",
    "ml",
    "no",
    "nn",
    "sk",
    "sl",
    "hr",
    "ro",
    "eu",
    "gl",
    "ka",
    "lv",
    "tl",
]


class JobSettings(BaseModel):
    task: TaskEnum = TaskEnum.TRANSCRIBE
    model: JobModelEnum = JobModelEnum.LARGE
    language: JobLangEnum | None = None  # None means automatic detection
    alignment: AlignmentSettings | None = AlignmentSettings()  # None means no alignment
    diarization: DiarizationSettings | None = None  # None means no diarization
    vad_settings: VadSettings = VadSettings()
    asr_settings: AsrSettings = AsrSettings()

    @model_validator(mode="after")
    def model_language_support_validation(self) -> Self:
        if self.language != JobLangEnum.ENGLISH and self.model in (
            JobModelEnum.TINY_EN,
            JobModelEnum.BASE_EN,
            JobModelEnum.SMALL_EN,
            JobModelEnum.MEDIUM_EN,
        ):
            raise ValueError(
                "If you want to use the 'tiny.en', 'base.en', 'small.en' or 'medium.en' models, you have to set the language to 'en'"
            )
        return self

    @model_validator(mode="after")
    def no_alignment_for_translation(self) -> Self:
        if self.task == TaskEnum.TRANSLATE and self.alignment is not None:
            raise ValueError("Alignment not supported for the translation task")
        return self

    @model_validator(mode="after")
    def alignment_supported_language(self) -> Self:
        if self.alignment is not None and self.language not in supported_alignment_languages:
            raise ValueError(
                f"language {self.language} is not supported for alignment. Either disable alignment or choose another language"
            )
        return self


class RunnerRegisterRequest(BaseModel):
    name: str = Field(max_length=40)
    version: str
    git_hash: str = Field(max_length=40)
    source_code_url: str
    priority: int = Field(gt=0)


class Transcript(BaseModel):
    as_txt: str
    as_srt: str
    as_tsv: str
    as_vtt: str
    as_json: dict


class RunnerSubmitResultRequest(BaseModel):
    error_msg: str | None = None
    transcript: Transcript | None = None

    @model_validator(mode="after")
    def either_error_or_transcript(self) -> Self:
        if self.error_msg is None and self.transcript is None:
            raise ValueError("Either `error_msg` or `transcript` must be set for job submission")
        return self


class HeartbeatRequest(BaseModel):
    progress: float = Field(
        ge=0.0,
        le=100.0,
        default=0.0,
    )
