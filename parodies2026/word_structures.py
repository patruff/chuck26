"""
Word structures for parody generation.

This module contains custom phonetic pronunciations, funny word lists,
and known funny parody examples for training/guidance.

Data is loaded from:
- known100.csv: 100 verified funny parody examples
- This file: Custom pronunciations and funny words list
"""

import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict

# =============================================================================
# FILE PATHS
# =============================================================================

# Get the directory where this module is located
MODULE_DIR = Path(__file__).parent

# Path to the known parodies CSV file
KNOWN_PARODIES_CSV = MODULE_DIR / "known100.csv"


# =============================================================================
# CUSTOM PHONETIC PRONUNCIATIONS
# =============================================================================
# For words not in CMU dictionary
# Format: {"word": {"primary_phones": "PRONUNCIATION"}}

custom_phones = {
    # Add custom pronunciations here as needed
    # Example:
    # "bitcoin": {"primary_phones": "B IH T K OY N"},
    "serbed": {"primary_phones": "S ER B D"},
    "codfather": {"primary_phones": "K AA D F AA DH ER"},
    "foodfellas": {"primary_phones": "F UW D F EH L AH Z"},
    "graveheart": {"primary_phones": "G R EY V HH AA R T"},
}


# =============================================================================
# KNOWN FUNNY PARODIES (loaded from CSV)
# =============================================================================
# These are verified funny parodies that work well. They serve as:
# 1. Examples for the model to learn from
# 2. Training data for RLVR
# 3. Style guidance for the type of humor we want


def load_known_parodies() -> List[Tuple[str, str, str]]:
    """
    Load known funny parodies from CSV file.

    Returns:
        List of tuples: (original_title, parody_title, reasoning)
    """
    parodies = []

    if not KNOWN_PARODIES_CSV.exists():
        print(f"Warning: {KNOWN_PARODIES_CSV} not found, using empty list")
        return parodies

    with open(KNOWN_PARODIES_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parodies.append((
                row['original'],
                row['parody'],
                row['reasoning']
            ))

    return parodies


# Load parodies at module import time
KNOWN_FUNNY_PARODIES = load_known_parodies()

# Convert to dict for quick lookup
KNOWN_PARODIES_DICT = {orig: parody for orig, parody, _ in KNOWN_FUNNY_PARODIES}


# =============================================================================
# FUNNY WORDS LIST
# =============================================================================
# These words are prioritized when looking for phonetic matches.
# They tend to create funnier parodies due to their comedic associations.
#
# Categories are provided for organization - the full list is used for matching.

FUNNY_WORDS_BY_CATEGORY = {
    "bodily_functions": [
        "fart", "poop", "butt", "turd", "pee", "snot", "booger", "barf", "puke",
        "burp", "belch", "gas", "dump", "crap", "diarrhea", "vomit", "spit",
    ],

    "adult_edgy": [
        "porn", "whore", "slut", "pimp", "hump", "grope", "fondle", "thrust",
        "moan", "groan", "climax", "orgy", "fetish", "kinky", "naughty", "booty",
    ],

    "violence_dark": [
        "kill", "murder", "stab", "choke", "strangle", "slaughter", "massacre",
        "gore", "blood", "death", "corpse", "grave", "tomb", "morgue", "slay",
    ],

    "silly_absurd": [
        "silly", "wacky", "crazy", "bonkers", "zany", "goofy", "dorky", "nerdy",
        "wonky", "funky", "chunky", "clunky", "junky", "flunky", "kooky", "loopy",
    ],

    "food": [
        "taco", "burrito", "sausage", "wiener", "pickle", "banana", "cucumber",
        "meatball", "noodle", "gravy", "sauce", "cheese", "bacon", "ham",
        "spam", "lard", "grease", "butter", "cream", "mayo", "toast", "waffle",
        "pancake", "biscuit", "nugget", "drumstick", "bologna", "salami",
    ],

    "animals": [
        "monkey", "donkey", "weasel", "ferret", "badger", "beaver", "otter",
        "llama", "alpaca", "platypus", "walrus", "manatee", "sloth", "wombat",
        "poodle", "pug", "chihuahua", "hamster", "gerbil", "squirrel",
    ],

    "body_parts": [
        "butt", "boob", "nipple", "belly", "navel", "armpit", "groin",
        "crotch", "buttock", "thigh", "rump", "rear", "bum", "tummy", "gut",
    ],

    "sounds_actions": [
        "splat", "splurt", "squirt", "squish", "squelch", "slurp", "gulp",
        "burp", "honk", "bonk", "clunk", "thunk", "whack", "smack", "thwack",
        "plop", "splosh", "whomp", "thud", "crack", "snap", "pop",
    ],

    "insults_light": [
        "dork", "nerd", "geek", "dweeb", "loser", "bozo", "doofus", "dimwit",
        "nitwit", "twit", "idiot", "moron", "fool", "dummy", "goofball",
        "numbskull", "blockhead", "bonehead", "meathead", "airhead",
    ],

    "textures_gross": [
        "moist", "crusty", "chunky", "lumpy", "soggy", "squishy", "gooey",
        "icky", "sticky", "slimy", "grimy", "grubby", "scruffy", "scrappy",
        "mushy", "gloppy", "gunky", "yucky", "oozy", "cruddy",
    ],

    "cultural_ethnic": [
        "serbian", "croatian", "polish", "swedish", "danish", "finnish",
        "scottish", "irish", "welsh", "bavarian", "prussian", "viking",
    ],

    "occupations": [
        "plumber", "janitor", "dentist", "proctologist", "mortician",
        "taxidermist", "podiatrist", "urologist", "chiropractor", "accountant",
    ],

    "misc_funny": [
        "underpants", "underwear", "panties", "diaper", "toilet", "potty",
        "booze", "drunk", "hangover", "bellyache", "hiccup", "sneeze",
        "toenail", "earwax", "dandruff", "fungus", "wart", "bunion",
    ],
}

# Flatten all categories into a single list for matching
funny_words = []
for category_words in FUNNY_WORDS_BY_CATEGORY.values():
    funny_words.extend(category_words)

# Remove duplicates while preserving order
funny_words = list(dict.fromkeys(funny_words))


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_example_prompt_text(num_examples: int = 10) -> str:
    """
    Generate example text for prompts showing known funny parodies.

    Args:
        num_examples: Number of examples to include (default: 10)

    Returns:
        Formatted string of examples
    """
    examples = []
    for orig, parody, explanation in KNOWN_FUNNY_PARODIES[:num_examples]:
        examples.append(f'  - "{orig}" → "{parody}" ({explanation})')
    return "\n".join(examples)


def get_parody_style_guide() -> str:
    """
    Return a style guide based on successful parodies.

    This is now imported from system_prompt.py for centralization,
    but kept here for backward compatibility.
    """
    try:
        from system_prompt import PARODY_STYLE_GUIDE
        return PARODY_STYLE_GUIDE
    except ImportError:
        # Fallback if system_prompt.py doesn't exist
        return """
PARODY STYLE GUIDE (based on successful examples):

1. PHONETIC PRIORITY: The parody must sound like the original when spoken
   - "You Got Served" → "You Got Serbed" (nearly identical pronunciation)
   - Single consonant/vowel swaps work best

2. HUMOR THROUGH CONTRAST: Subvert expectations
   - Epic → Mundane: "The Matrix" → "The Mattress"
   - Serious → Silly: "Die Hard" → "Dye Hard"
   - Action → Domestic: "Kill Bill" → "Pill Bill"

3. EDGY IS GOOD: Don't shy away from adult themes
   - "Star Wars" → "Scar Whores" (pushes boundaries)
   - "The Office" → "The Orifice" (crude but funny)

4. COMPOUND WORDS: Replace parts while keeping structure
   - "The Godfather" → "The Codfather" (god→cod)
   - "Goodfellas" → "Foodfellas" (good→food)

5. DOUBLE CHANGES: Sometimes changing 2 words is funnier
   - "Pulp Fiction" → "Gulp Friction" (both words changed)
   - Creates completely new absurd meaning
"""


def get_words_by_category(category: str) -> List[str]:
    """
    Get funny words from a specific category.

    Args:
        category: Category name (e.g., 'food', 'animals', 'adult_edgy')

    Returns:
        List of words in that category
    """
    return FUNNY_WORDS_BY_CATEGORY.get(category, [])


def get_all_categories() -> List[str]:
    """Return list of all available funny word categories."""
    return list(FUNNY_WORDS_BY_CATEGORY.keys())


def search_parodies(query: str) -> List[Tuple[str, str, str]]:
    """
    Search known parodies by original title.

    Args:
        query: Search string (case-insensitive)

    Returns:
        List of matching (original, parody, reasoning) tuples
    """
    query_lower = query.lower()
    return [
        (orig, parody, reason)
        for orig, parody, reason in KNOWN_FUNNY_PARODIES
        if query_lower in orig.lower()
    ]


def get_parody_for_title(title: str) -> str | None:
    """
    Get the known parody for a specific title if it exists.

    Args:
        title: Original title to look up

    Returns:
        Parody title if found, None otherwise
    """
    return KNOWN_PARODIES_DICT.get(title)


# =============================================================================
# MODULE INFO
# =============================================================================

def print_stats():
    """Print statistics about the loaded data."""
    print(f"Known Parodies: {len(KNOWN_FUNNY_PARODIES)}")
    print(f"Funny Words: {len(funny_words)}")
    print(f"Word Categories: {len(FUNNY_WORDS_BY_CATEGORY)}")
    print(f"\nCategories: {', '.join(get_all_categories())}")


if __name__ == "__main__":
    print_stats()
    print("\nSample parodies:")
    for orig, parody, reason in KNOWN_FUNNY_PARODIES[:5]:
        print(f"  {orig} → {parody}")
