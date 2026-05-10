"""Application constants: categories, API defaults, image limits."""

from __future__ import annotations

import os
from enum import Enum

# Canonical output folders used by sorter in strict mode.
CANONICAL_CATEGORIES: tuple[str, ...] = (
    "explicit_zoo_real_animal",
    "furry_nsfw_canidae",
    "furry_nsfw_other",
    "furry_sfw_canidae",
    "furry_sfw_other",
    "human_real_nsfw_female",
    "human_real_nsfw_male",
    "human_real_sfw",
    "human_ai_gen_nsfw_female",
    "human_ai_gen_nsfw_male",
    "human_ai_gen_sfw",
    "personal_user_sfw",
    "personal_user_nsfw",
    "my_dog",
    "puppy_play",
    "real_animals",
    "vehicles_and_racing",
    "memes_and_screenshots",
    "landscapes_and_objects",
    "uncategorized",
)

# Strict category list (exact tag strings for model output).
# First block is canonical folders, second block keeps legacy compatibility tags.
CATEGORIES: tuple[str, ...] = (
    *CANONICAL_CATEGORIES,
    # Legacy tags (kept for backwards compatibility and prompt understanding)
    "human_nsfw_solo_male",
    "human_nsfw_solo_female",
    "human_nsfw_group",
    "human_sfw",
    "cars_and_bmw",
)

CATEGORY_WHITELIST: frozenset[str] = frozenset(CATEGORIES)
CANONICAL_CATEGORY_WHITELIST: frozenset[str] = frozenset(CANONICAL_CATEGORIES)

UNCATEGORIZED = "uncategorized"

# Правила приоритета при конфликте триггеров (вставляется в системный промпт целиком)
PRIORITY_RULES_BLOCK: str = """
PRIORITY RESOLUTION (mandatory). If several categories could apply, output exactly ONE tag using this order (highest first):

PRIORITY 1 — Owner, dog, fetish (prefer the first that truly matches):
  personal_user_nsfw → personal_user_sfw → my_dog → puppy_play

PRIORITY 2 — Furry content (NSFW before SFW when both apply): furry_nsfw_canidae, furry_nsfw_other, then furry_sfw_canidae, furry_sfw_other

PRIORITY 3 — Human photo NSFW: human_real_nsfw_male, human_real_nsfw_female

PRIORITY 4 — Human AI-generated NSFW: human_ai_gen_nsfw_male, human_ai_gen_nsfw_female

PRIORITY 5 — Explicit zoophilia with real animal: explicit_zoo_real_animal

PRIORITY 6 — Vehicles: vehicles_and_racing

PRIORITY 7 — Everything else: human_real_sfw, human_ai_gen_sfw, real_animals, memes_and_screenshots, landscapes_and_objects, or uncategorized as appropriate.
""".strip()

# Описание для каждого тега (редактируйте здесь)
CATEGORY_PROMPTS: dict[str, str] = {
    "personal_user_sfw": (
        "The app owner, safe-for-work only. Match USER_CONTEXT (e.g. male, 195cm, blonde hair, blue eyes). "
        "No nudity or explicit sexual content."
    ),
    "personal_user_nsfw": (
        "The app owner, NSFW or nudity. Strong trigger: black wolf paw with forest tattoo on the right side of the stomach; "
        "also match USER_CONTEXT when clearly the same person."
    ),
    "my_dog": (
        "The owner's specific dog. Match USER_CONTEXT: Black Labrador. Not generic real_animals."
    ),
    "puppy_play": (
        "Puppy play / pet play kink: gear or roleplay (hoods, collars, bone tags, etc.), any intensity; use when theme is central."
    ),
    "furry_nsfw_canidae": (
        "NSFW anthropomorphic art: clearly canine (wolf, dog, fox, etc.). Explicit sexual content."
    ),
    "furry_nsfw_other": (
        "NSFW anthropomorphic art: non-canine species (dragons, cats, birds, etc.). Explicit sexual content."
    ),
    "furry_sfw_canidae": (
        "SFW furry art: canine characters; no explicit sexual content."
    ),
    "furry_sfw_other": (
        "SFW furry art: non-canine characters; no explicit sexual content."
    ),
    "human_real_nsfw_female": (
        "Photorealistic real human female, NSFW or nudity. Not the app owner."
    ),
    "human_real_nsfw_male": (
        "Photorealistic real human male, NSFW or nudity. Not the app owner."
    ),
    "human_real_sfw": (
        "Photorealistic real humans (any gender), SFW. Not the app owner."
    ),
    "human_ai_gen_nsfw_female": (
        "AI-generated female human, NSFW or nudity."
    ),
    "human_ai_gen_nsfw_male": (
        "AI-generated male human, NSFW or nudity."
    ),
    "human_ai_gen_sfw": (
        "AI-generated human, SFW."
    ),
    "explicit_zoo_real_animal": (
        "Explicit NSFW scene involving a real animal and human (zoophilia context)."
    ),
    "vehicles_and_racing": (
        "Vehicles and racing: cars, bikes, motorsport, driving POV, car meets."
    ),
    "real_animals": (
        "Real animals in photos; not anthropomorphic art; not the Black Labrador (my_dog)."
    ),
    "memes_and_screenshots": (
        "Text-heavy images, UI screenshots, memes, social media captures."
    ),
    "landscapes_and_objects": (
        "Dark aesthetics, neon, empty rooms, landscapes, inanimate objects as main subject."
    ),
    "uncategorized": (
        "Ambiguous, severe visual noise, or does not fit any tag clearly."
    ),
    # Legacy-compatible descriptions
    "human_nsfw_solo_male": (
        "Legacy alias for human_real_nsfw_male."
    ),
    "human_nsfw_solo_female": (
        "Legacy alias for human_real_nsfw_female."
    ),
    "human_nsfw_group": (
        "Legacy alias for human_real_nsfw_female or human_real_nsfw_male depending on dominant subject."
    ),
    "human_sfw": (
        "Legacy alias for human_real_sfw."
    ),
    "cars_and_bmw": (
        "Legacy alias for vehicles_and_racing."
    ),
}

# Статичные изображения (без .gif — см. режим сканирования)
STILL_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
        ".tif",
        ".tiff",
        ".heic",
        ".heif",
        ".avif",
    }
)

# Видеоконтейнеры (не GIF)
VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".mp4",
        ".mov",
        ".mkv",
        ".webm",
        ".avi",
        ".m4v",
        ".wmv",
        ".flv",
        ".mpeg",
        ".mpg",
        ".3gp",
        ".mts",
        ".m2ts",
    }
)

GIF_EXTENSION = ".gif"

# Обратная совместимость: всё, что считаем «фото» для whitelist расширений в старом коде
IMAGE_EXTENSIONS: frozenset[str] = STILL_IMAGE_EXTENSIONS

VIDEO_FRAME_COUNT = 3

# ffmpeg: папка ffmpeg-runtime/bin рядом с проектом (run.bat добавляет в PATH) или
# PHOTO_AI_SORTER_FFMPEG / PHOTO_AI_SORTER_FFPROBE — полный путь при необходимости.

# Доли длительности (0..1) для выборочных кадров — без полного прогона видео
VIDEO_SAMPLE_FRACTIONS: tuple[float, ...] = (0.0, 0.5, 1.0)

# Окно декода после seek: ffmpeg читает только короткий отрезок, не весь файл
VIDEO_FRAGMENT_DECODE_SEC = 0.15

# Таймаут одного вызова ffmpeg на кадр (сек)
FFMPEG_FRAME_TIMEOUT_SEC = 45.0

# Версия пайплайна (категории / кадры / промпт): смена → переобработка в БД
PIPELINE_VERSION = "2026-05-10-categories-v3"

# Запас свободного места при копировании (байт)
COPY_FREE_MARGIN_BYTES = 64 * 1024 * 1024

# Порядок слияния при конфликте тегов с разных кадров (меньший индекс = выигрывает)
TAG_MERGE_PRIORITY: tuple[str, ...] = (
    "personal_user_nsfw",
    "personal_user_sfw",
    "my_dog",
    "puppy_play",
    "explicit_zoo_real_animal",
    "furry_nsfw_canidae",
    "furry_nsfw_other",
    "furry_sfw_canidae",
    "furry_sfw_other",
    "human_real_nsfw_male",
    "human_real_nsfw_female",
    "human_ai_gen_nsfw_male",
    "human_ai_gen_nsfw_female",
    "vehicles_and_racing",
    "human_real_sfw",
    "human_ai_gen_sfw",
    "real_animals",
    "memes_and_screenshots",
    "landscapes_and_objects",
    "uncategorized",
)


class MediaScanMode(str, Enum):
    """Режим обхода папки: только фото / фото+видео / только видео+GIF."""

    PHOTOS_ONLY = "photos_only"
    PHOTOS_AND_VIDEO = "photos_and_video"
    VIDEO_ONLY = "video_only"

DEFAULT_API_BASE = "http://10.77.77.2:29931"
# Дефолтный ключ в коде намеренно; PHOTO_AI_SORTER_API_KEY в env перекрывает его.
DEFAULT_API_KEY = os.environ.get(
    "PHOTO_AI_SORTER_API_KEY",
    "sk-lm-GYikKNMF:U63StF1eS2aT7o0b3HEl",
)
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"

MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85

REQUEST_CONNECT_TIMEOUT_SEC = 30.0
REQUEST_READ_TIMEOUT_SEC = 600.0
REQUEST_TIMEOUT_SEC = 600.0

API_MAX_RETRIES = 3
API_RETRY_BACKOFF_SEC: tuple[float, ...] = (2.0, 5.0)

API_PROBE_TIMEOUT_SEC = 15
VISION_TEST_TIMEOUT_SEC = 90

CHAT_COMPLETION_MAX_TOKENS = 1024
VISION_PROBE_MAX_TOKENS = 512

DEFAULT_MODEL = "local-model"

LOG_MAX_LINES = 500

PARALLEL_WORKERS = 3

ETA_ROLLING_WINDOW = 20

# Повторная попытка классификации файла при API-ошибке или uncategorized.
CLASSIFY_FILE_MAX_ATTEMPTS = 3
