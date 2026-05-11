"""Application constants: categories, API defaults, image limits."""

from __future__ import annotations

import os
from enum import Enum

from app.local_env import load_project_env

load_project_env()

# Canonical output folders for the «furry» preset (strict tag mode).
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

# «Furry» preset: exact tag strings for strict mode (canonical + legacy aliases).
CATEGORIES: tuple[str, ...] = (
    *CANONICAL_CATEGORIES,
    # Legacy tags (kept for backwards compatibility and prompt understanding)
    "human_nsfw_solo_male",
    "human_nsfw_solo_female",
    "human_nsfw_group",
    "human_sfw",
    "cars_and_bmw",
)

# «Общий» preset: furry tags plus extra buckets for geek / IT / car culture / nightlife / LGBTQ+ galleries.
GENERAL_EXTRA_CATEGORIES: tuple[str, ...] = (
    "tech_desk_setup",
    "pc_build_and_hardware",
    "coding_ide_and_terminal",
    "gaming_ui_screenshots",
    "gaming_room_setup",
    "anime_and_manga",
    "comics_and_superheroes",
    "board_games_tabletop",
    "sneakers_and_streetwear",
    "gym_and_fitness",
    "car_mods_and_meets",
    "nightlife_party",
    "coffee_and_food_aesthetic",
    "music_festival_live",
    "travel_urban_explore",
    "pride_and_lgbt_events",
    "gay_male_nsfw_solo",
    "gay_male_nsfw_couple",
    "queer_art_sfw",
    "tattoos_and_body_art",
    "streaming_and_webcam",
    "sci_fi_collectibles",
)

GENERAL_CATEGORIES: tuple[str, ...] = (*CATEGORIES, *GENERAL_EXTRA_CATEGORIES)

FURRY_CATEGORY_WHITELIST: frozenset[str] = frozenset(CATEGORIES)
GENERAL_CATEGORY_WHITELIST: frozenset[str] = frozenset(GENERAL_CATEGORIES)

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
        "The app owner, safe-for-work only. Match USER_CONTEXT description. "
        "No nudity or explicit sexual content."
    ),
    "personal_user_nsfw": (
        "The app owner, NSFW or nudity. Match USER_CONTEXT description "
        "when clearly the same person."
    ),
    "my_dog": (
        "The owner's specific dog/pet. Match USER_CONTEXT description. Not generic real_animals."
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
        "Real animals in photos; not anthropomorphic art; not the owner's pet (my_dog, see USER_CONTEXT)."
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
    # --- «Общий» preset: extra categories (after furry list) ---
    "tech_desk_setup": (
        "Desk / battlestation: monitors, peripherals, cable management, home office rig (not a full PC build teardown)."
    ),
    "pc_build_and_hardware": (
        "PC internals and hardware focus: GPUs, motherboards, cooling, benchmarks, component boxes."
    ),
    "coding_ide_and_terminal": (
        "IDE, code editor, stack traces, terminal windows, git CLI, config files as main subject."
    ),
    "gaming_ui_screenshots": (
        "In-game UI, HUD, menus, achievements, match results; gameplay screen as primary content."
    ),
    "gaming_room_setup": (
        "Gaming space: consoles, RGB rigs, posters, shelves of games; not only a generic desk photo."
    ),
    "anime_and_manga": (
        "Anime / manga art, covers, figures posed as collectibles, cosplay clearly referencing 2D franchises."
    ),
    "comics_and_superheroes": (
        "Western comics, superhero merch, comic con cosplay (non-anime), graphic novel art."
    ),
    "board_games_tabletop": (
        "Board games, RPG tables, dice, miniatures battle scenes, card games in play."
    ),
    "sneakers_and_streetwear": (
        "Sneakers, streetwear fits, hype drops, outfit flatlays where footwear or urban fashion dominates."
    ),
    "gym_and_fitness": (
        "Gym training, progress pics, sportswear, weights and machines as main subject."
    ),
    "car_mods_and_meets": (
        "Modified cars, stance, wraps, car-meet culture; use when tuning/meet vibe matters more than generic driving."
    ),
    "nightlife_party": (
        "Club, party, bar night, dance floor lighting, social night-out photos."
    ),
    "coffee_and_food_aesthetic": (
        "Café flatlays, brunch, craft drinks, food styling as main subject."
    ),
    "music_festival_live": (
        "Concerts, festivals, stage lights, crowd from live music events."
    ),
    "travel_urban_explore": (
        "City travel, street exploration, architecture walks; not pure landscape wallpaper."
    ),
    "pride_and_lgbt_events": (
        "Pride parades, rainbow flags, LGBT community events, protest or celebration context."
    ),
    "gay_male_nsfw_solo": (
        "NSFW male-presenting solo erotic content; not the app owner unless USER_CONTEXT matches."
    ),
    "gay_male_nsfw_couple": (
        "NSFW male+male couple erotic content; not the app owner unless USER_CONTEXT matches."
    ),
    "queer_art_sfw": (
        "SFW queer-themed illustration or photography without explicit sex acts."
    ),
    "tattoos_and_body_art": (
        "Tattoos, piercings, body art close-ups as the primary subject."
    ),
    "streaming_and_webcam": (
        "Stream overlay, chat on screen, OBS layout, webcam framing for content creators."
    ),
    "sci_fi_collectibles": (
        "Sci-fi models, props, franchise collectibles (non-anime), helmets, replicas."
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
PIPELINE_VERSION = "2026-05-11-general-preset-v5"

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
    "gay_male_nsfw_solo",
    "gay_male_nsfw_couple",
    "vehicles_and_racing",
    "human_real_sfw",
    "human_ai_gen_sfw",
    "real_animals",
    "memes_and_screenshots",
    "landscapes_and_objects",
    "queer_art_sfw",
    "pride_and_lgbt_events",
    "tech_desk_setup",
    "pc_build_and_hardware",
    "coding_ide_and_terminal",
    "gaming_ui_screenshots",
    "gaming_room_setup",
    "anime_and_manga",
    "comics_and_superheroes",
    "board_games_tabletop",
    "sneakers_and_streetwear",
    "gym_and_fitness",
    "car_mods_and_meets",
    "nightlife_party",
    "coffee_and_food_aesthetic",
    "music_festival_live",
    "travel_urban_explore",
    "tattoos_and_body_art",
    "streaming_and_webcam",
    "sci_fi_collectibles",
    "uncategorized",
)


class MediaScanMode(str, Enum):
    """Режим обхода папки: только фото / фото+видео / только видео+GIF."""

    PHOTOS_ONLY = "photos_only"
    PHOTOS_AND_VIDEO = "photos_and_video"
    VIDEO_ONLY = "video_only"

DEFAULT_API_BASE = (
    os.environ.get("PHOTO_AI_SORTER_API_BASE", "http://127.0.0.1:1234").strip()
    or "http://127.0.0.1:1234"
)
# LM Studio usually accepts requests without a real API key. If your OpenAI-compatible
# server requires one, provide it through the environment instead of committing it.
DEFAULT_API_KEY = os.environ.get("PHOTO_AI_SORTER_API_KEY", "")
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"
MODELS_PATH = "/v1/models"

MAX_IMAGE_SIDE = 1024
JPEG_QUALITY = 85

REQUEST_CONNECT_TIMEOUT_SEC = 30.0
REQUEST_READ_TIMEOUT_SEC = 600.0
API_MAX_RETRIES = 5
API_RETRY_BACKOFF_SEC: tuple[float, ...] = (1.5, 3.0, 5.0, 8.0)

API_PROBE_TIMEOUT_SEC = 15
VISION_TEST_TIMEOUT_SEC = 90

CHAT_COMPLETION_MAX_TOKENS = 1024
VISION_PROBE_MAX_TOKENS = 512

DEFAULT_MODEL = os.environ.get("PHOTO_AI_SORTER_MODEL", "local-model").strip() or "local-model"

LOG_MAX_LINES = 500

ETA_ROLLING_WINDOW = 20

# Повторная попытка классификации файла при API-ошибке или uncategorized.
CLASSIFY_FILE_MAX_ATTEMPTS = 3
