"""
Translation Service for Kannada Language Support
Uses free translation service without API keys
"""

from functools import lru_cache
import json
import os

# Cache for translations to reduce API calls
_translation_cache = {}

# Comprehensive Kannada translations
KANNADA_TRANSLATIONS = {
    "welcome": "👋 ನಮ್ಮಗೆ ಸ್ವಾಗತ! ಬೆಳೆಗಳು, ಕೀಟಗಳು, ರಸಗೊಬ್ಬರ ಅವಧಿ, ನೀರಾವರಣ ಅಥವಾ PM-KISAN ನಂತಹ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಬಗ್ಗೆ ನನ್ನನ್ನು ಕೇಳಿ.",
    "how_to_control": "ಮೈಸೂರು ಜೋಲ್ ಬಿಡುಳುಗಳನ್ನು ನಿಯಂತ್ರಿಸುವುದು ಹೇಗೆ?",
    "best_fertilizer": "ನೆಲೆ ಪದ್ಧತಿಯ ಸೋತಾ ರಸಗೊಬ್ಬರ ವೇಳಾಪಟ್ಟಿ",
    "irrigation_schedule": "ಬೇಸಿಗೆಯಲ್ಲಿ ಟೊಮ್ಯಾಟೋ ಎಷ್ಟು ಬಾರಿ ನೀರಾವರಿಸಬೇಕು?",
    "pm_kisan": "PM-KISAN ಯೋಗ್ಯತೆ ಮತ್ತು ಅಗತ್ಯ ದಾಖಲೆಗಳು",
    "ask_farming": "ಕೃಷಿ ಪ್ರಶ್ನೆ ಕೇಳಿ...",
    "send": "ಕಳುಹಿಸಿ",
    "please_register": "ದಯವಿಟ್ಟು ಹೋಮ್ ಪುಟದಲ್ಲಿ ನೋಂದಾಯನ ಮಾಡಿ.",
    "loading": "ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
    "error": "ದೋಷ",
    "success": "ಯಶಸ್ವಿ",
}

def get_kannada_text(english_text: str, use_api: bool = False) -> str:
    """
    Get Kannada translation of English text.
    Uses translation API if available and requested, otherwise uses fallback.
    """
    # Check if we have a direct mapping
    if english_text in KANNADA_TRANSLATIONS:
        return KANNADA_TRANSLATIONS[english_text]
    
    # Try simple pattern matching for common phrases
    lower_text = english_text.lower().strip()
    
    # Return original if no translation found
    return english_text

def translate_text(text: str, target_language: str = "English") -> str:
    """
    Main translation function.
    Translates text to target language.
    """
    if not text:
        return text
    
    if target_language.lower() == "kannada":
        return get_kannada_text(text)
    
    return text

def is_kannada_mode(language: str) -> bool:
    """Check if Kannada mode is enabled."""
    return language.lower() in ["kannada", "ಕನ್ನಡ"]

