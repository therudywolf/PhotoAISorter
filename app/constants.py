# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Photo AI Sorter contributors — see NOTICE

"""Application constants: categories, search profiles, API defaults, image limits."""

from __future__ import annotations

import os
from enum import Enum

from app.local_env import load_project_env

load_project_env()


# ---------------------------------------------------------------------------
# Search profiles: combine content-safety and style filters
# ---------------------------------------------------------------------------

class SearchProfile(str, Enum):
    """Pre-built tag set profiles for the fixed-list sort modes."""
    SFW = "sfw"
    NSFW = "nsfw"
    FURRY_SFW = "furry_sfw"
    FURRY_NSFW = "furry_nsfw"


# ---------------------------------------------------------------------------
# Base SFW categories (included in ALL profiles)
# ---------------------------------------------------------------------------

BASE_SFW_CATEGORIES: tuple[str, ...] = (
    "humans_sfw",
    "ai_generated_sfw",
    "real_animals",
    "vehicles_and_racing",
    "memes_and_screenshots",
    "landscapes_and_objects",
    "uncategorized",
)

# ---------------------------------------------------------------------------
# NSFW addon categories (included when profile allows NSFW)
# ---------------------------------------------------------------------------

NSFW_ADDON_CATEGORIES: tuple[str, ...] = (
    "humans_nsfw_female",
    "humans_nsfw_male",
    "ai_generated_nsfw_female",
    "ai_generated_nsfw_male",
)

# ---------------------------------------------------------------------------
# Furry addon categories
# ---------------------------------------------------------------------------

FURRY_SFW_ADDON_CATEGORIES: tuple[str, ...] = (
    "furry_sfw_canidae",
    "furry_sfw_other",
)

FURRY_NSFW_ADDON_CATEGORIES: tuple[str, ...] = (
    "furry_nsfw_canidae",
    "furry_nsfw_other",
)

# ---------------------------------------------------------------------------
# Extended general categories (tech, hobby, lifestyle — always SFW)
# ---------------------------------------------------------------------------

EXTENDED_CATEGORIES: tuple[str, ...] = (
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
)


# ---------------------------------------------------------------------------
# Profile → category assembly
# ---------------------------------------------------------------------------

def categories_for_profile(profile: SearchProfile) -> tuple[str, ...]:
    """Assemble the full category list for a given search profile."""
    cats: list[str] = list(BASE_SFW_CATEGORIES[:-1])  # exclude uncategorized (add last)
    cats.extend(EXTENDED_CATEGORIES)

    if profile in (SearchProfile.NSFW, SearchProfile.FURRY_NSFW):
        cats.extend(NSFW_ADDON_CATEGORIES)

    if profile in (SearchProfile.FURRY_SFW, SearchProfile.FURRY_NSFW):
        cats.extend(FURRY_SFW_ADDON_CATEGORIES)

    if profile == SearchProfile.FURRY_NSFW:
        cats.extend(FURRY_NSFW_ADDON_CATEGORIES)

    cats.append("uncategorized")
    return tuple(cats)


# Precomputed category tuples for each profile
SFW_CATEGORIES: tuple[str, ...] = categories_for_profile(SearchProfile.SFW)
NSFW_CATEGORIES: tuple[str, ...] = categories_for_profile(SearchProfile.NSFW)
FURRY_SFW_CATEGORIES: tuple[str, ...] = categories_for_profile(SearchProfile.FURRY_SFW)
FURRY_NSFW_CATEGORIES: tuple[str, ...] = categories_for_profile(SearchProfile.FURRY_NSFW)

# Default categories used when no profile is specified
CATEGORIES: tuple[str, ...] = NSFW_CATEGORIES
GENERAL_CATEGORIES: tuple[str, ...] = FURRY_NSFW_CATEGORIES

# Whitelists (frozen sets for fast lookup)
CANONICAL_CATEGORIES: tuple[str, ...] = SFW_CATEGORIES
CANONICAL_CATEGORY_WHITELIST: frozenset[str] = frozenset(SFW_CATEGORIES)
GENERAL_CATEGORY_WHITELIST: frozenset[str] = frozenset(FURRY_NSFW_CATEGORIES)
FURRY_CATEGORY_WHITELIST: frozenset[str] = frozenset(FURRY_NSFW_CATEGORIES)

UNCATEGORIZED = "uncategorized"


# ---------------------------------------------------------------------------
# Priority rules (generic — no personal references)
# ---------------------------------------------------------------------------

PRIORITY_RULES_BLOCK: str = """
PRIORITY RESOLUTION (mandatory). If several categories could apply, output exactly ONE tag using this order (highest first):

PRIORITY 1 — Furry NSFW (if applicable): furry_nsfw_canidae, furry_nsfw_other

PRIORITY 2 — Furry SFW: furry_sfw_canidae, furry_sfw_other

PRIORITY 3 — Human NSFW: humans_nsfw_male, humans_nsfw_female

PRIORITY 4 — AI-generated NSFW: ai_generated_nsfw_male, ai_generated_nsfw_female

PRIORITY 5 — Vehicles: vehicles_and_racing

PRIORITY 6 — Everything else: humans_sfw, ai_generated_sfw, real_animals, memes_and_screenshots, landscapes_and_objects, or uncategorized as appropriate.
""".strip()


# ---------------------------------------------------------------------------
# Category descriptions (for the LLM system prompt)
# ---------------------------------------------------------------------------

CATEGORY_PROMPTS: dict[str, str] = {
    "humans_sfw": (
        "Real or AI-generated humans, safe-for-work. No nudity or explicit content."
    ),
    "humans_nsfw_female": (
        "Real human female, NSFW or nudity."
    ),
    "humans_nsfw_male": (
        "Real human male, NSFW or nudity."
    ),
    "ai_generated_sfw": (
        "AI-generated image, SFW. Digital art, renders, or neural network outputs without explicit content."
    ),
    "ai_generated_nsfw_female": (
        "AI-generated female, NSFW or nudity."
    ),
    "ai_generated_nsfw_male": (
        "AI-generated male, NSFW or nudity."
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
    "vehicles_and_racing": (
        "Vehicles and racing: cars, bikes, motorsport, driving POV, car meets."
    ),
    "real_animals": (
        "Real animals in photos; not anthropomorphic art."
    ),
    "memes_and_screenshots": (
        "Text-heavy images, UI screenshots, memes, social media captures."
    ),
    "landscapes_and_objects": (
        "Landscapes, architecture, inanimate objects, abstract backgrounds."
    ),
    "uncategorized": (
        "Ambiguous, severe visual noise, or does not fit any tag clearly."
    ),
    # Extended categories
    "tech_desk_setup": (
        "Desk / battlestation: monitors, peripherals, cable management, home office rig."
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
        "Gaming space: consoles, RGB rigs, posters, shelves of games."
    ),
    "anime_and_manga": (
        "Anime / manga art, covers, figures posed as collectibles, cosplay referencing 2D franchises."
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
        "Modified cars, stance, wraps, car-meet culture; tuning/meet vibe rather than generic driving."
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


# ---------------------------------------------------------------------------
# Tag merge priority (for multi-frame video classification)
# ---------------------------------------------------------------------------

TAG_MERGE_PRIORITY: tuple[str, ...] = (
    "furry_nsfw_canidae",
    "furry_nsfw_other",
    "furry_sfw_canidae",
    "furry_sfw_other",
    "humans_nsfw_male",
    "humans_nsfw_female",
    "ai_generated_nsfw_male",
    "ai_generated_nsfw_female",
    "vehicles_and_racing",
    "humans_sfw",
    "ai_generated_sfw",
    "real_animals",
    "memes_and_screenshots",
    "landscapes_and_objects",
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


# ---------------------------------------------------------------------------
# Media file extensions
# ---------------------------------------------------------------------------

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

VIDEO_SAMPLE_FRACTIONS: tuple[float, ...] = (0.0, 0.5, 1.0)
VIDEO_FRAGMENT_DECODE_SEC = 0.15
FFMPEG_FRAME_TIMEOUT_SEC = 45.0

PIPELINE_VERSION = "2026-05-22-hybrid-clip-v6-strict"

COPY_FREE_MARGIN_BYTES = 64 * 1024 * 1024


class MediaScanMode(str, Enum):
    """Scan mode: photos only / photos+video / video+GIF only."""

    PHOTOS_ONLY = "photos_only"
    PHOTOS_AND_VIDEO = "photos_and_video"
    VIDEO_ONLY = "video_only"


# ---------------------------------------------------------------------------
# API / network defaults
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = (
    os.environ.get("PHOTO_AI_SORTER_API_BASE", "http://127.0.0.1:1234").strip()
    or "http://127.0.0.1:1234"
)
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

CLASSIFY_FILE_MAX_ATTEMPTS = 3
