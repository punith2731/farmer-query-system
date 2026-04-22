import hashlib
import json
import os
import re
import base64
from datetime import datetime, timedelta
from typing import Optional
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
from price_forecasting import CropPriceForecastingEngine
from rag.qa_chain import ask
from voice.speech_to_text import transcribe
from voice.text_to_speech import speak_to_bytes

load_dotenv()

# ── Query validator ────────────────────────────────────────────────────────
_FARMING_KEYWORDS = re.compile(
    r"crop|plant|seed|soil|fertilizer|fertilis|pesticide|pest|insect|worm|disease|"
    r"irrigat|water|rain|drought|harvest|yield|paddy|rice|wheat|maize|corn|cotton|"
    r"soybean|tomato|onion|potato|sugarcane|groundnut|pulses|vegetable|fruit|farm|"
    r"agricultur|kisan|pm.kisan|subsidy|scheme|government|organic|compost|manure|"
    r"spray|fungicide|herbicide|weed|nitrogen|phosphorus|potassium|npk|urea|"
    r"sowing|planting|pruning|tilling|tillage|blight|rot|mildew|aphid|thrip|"
    r"nematode|borers|armyworm|locust|cultivation|horticulture|floriculture",
    re.IGNORECASE,
)
_MIN_WORDS = 2
_MAX_WORDS = 300
_GIBBERISH = re.compile(r"^[^a-zA-Z0-9\s]{3,}$|^(.)\1{4,}$")


def _extract_main_keyword(text: str) -> str:
    """Extract the most relevant farming keyword from the text."""
    top_keywords = [
        "crop", "plant", "seed", "soil", "fertilizer", "pesticide", "pest", "disease",
        "irrigation", "water", "harvest", "yield", "paddy", "rice", "wheat", "maize",
        "tomato", "onion", "potato", "cotton", "soybean", "groundnut",
        "aphid", "blight", "mildew", "nitrogen", "phosphorus", "potassium",
        "sowing", "planting", "pruning", "weed", "fungicide", "herbicide",
        "armyworm", "locust", "pm-kisan", "subsidy", "scheme"
    ]
    text_lower = text.lower()
    for keyword in top_keywords:
        if re.search(r"\b" + keyword + r"\b", text_lower, re.IGNORECASE):
            return keyword.capitalize()
    # Fallback: extract first farming keyword found
    match = _FARMING_KEYWORDS.search(text)
    return match.group(0).capitalize() if match else "Farming Tips"


def _expand_farmer_question(text: str, language: str = "English") -> str:
    """Return a natural-language expansion of the user's question for display."""
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""

    cleaned = cleaned[0].upper() + cleaned[1:] if len(cleaned) > 1 else cleaned.upper()
    if not cleaned.endswith(("?", ".", "!")):
        cleaned += "?"

    if language == "Kannada":
        return (
            "ನೀವು ಪ್ರಾಯೋಗಿಕ ಕೃಷಿ ಮಾರ್ಗದರ್ಶನವನ್ನು ಕೇಳುತ್ತಿದ್ದೀರಿ: "
            f"{cleaned} ಇದನ್ನು ಸ್ಪಷ್ಟವಾಗಿ ಮತ್ತು ರೈತರಿಗೆ ಸುಲಭವಾಗಿ ಅರ್ಥವಾಗುವಂತೆ ನೋಡೋಣ."
        )

    return (
        "You are asking for practical guidance about "
        f"{cleaned.lower()} Let us break this down in a clear, farmer-friendly way."
    )


def _is_valid_query(text: str) -> tuple[bool, str]:
    """Return (is_valid, reason). Checks length, gibberish, and topic relevance."""
    text = text.strip()
    if not text:
        return False, "empty"
    words = text.split()
    if len(words) < _MIN_WORDS:
        return False, "too short"
    if len(words) > _MAX_WORDS:
        return False, "too long"
    if _GIBBERISH.search(text):
        return False, "gibberish"
    if not _FARMING_KEYWORDS.search(text):
        return False, "off-topic"
    return True, "ok"


def _get_invalid_response(reason: str, language: str = "English") -> str:
    """Get language-specific error message for invalid queries."""
    invalid_responses = {
        "English": {
            "empty": "⚠️ Please type a question before sending.",
            "too short": "⚠️ Your query is too short. Please ask a complete farming-related question.",
            "too long": "⚠️ Your message is too long. Please shorten your question.",
            "gibberish": "❌ That doesn't look like a valid question. Please ask something about farming, crops, or agriculture.",
            "off-topic": "❌ **Invalid query.** I can only answer questions related to farming, crops, pests, fertilizers, irrigation, or government agricultural schemes like PM-KISAN. Please rephrase your question.",
        },
        "Kannada": {
            "empty": "⚠️ ಕಳುಹಿಸುವ ಮೊದಲು ದಯವಿಟ್ಟು ಒಂದು ಪ್ರಶ್ನೆ ಟೈಪ್ ಮಾಡಿ.",
            "too short": "⚠️ ನಿಮ್ಮ ಪ್ರಶ್ನೆ ತುಂಬಾ ಚಿಕ್ಕದಾಗಿದೆ. ದಯವಿಟ್ಟು ಸಂಪೂರ್ಣ ಕೃಷಿ-ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ.",
            "too long": "⚠️ ನಿಮ್ಮ ಸಂದೇಶ ತುಂಬಾ ಉದ್ದವಾಗಿದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಸಂಕ್ಷಿಪ್ತಗೊಳಿಸಿ.",
            "gibberish": "❌ ಇದು ಸಿಂಧುವಾಗಿ ತೋರುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಕೃಷಿ, ಬೆಳೆಗಳು ಅಥವಾ ಕೃಷಿಯ ಬಗ್ಗೆ ಏನನ್ನಾದರೂ ಕೇಳಿ.",
            "off-topic": "❌ **ಅಮಾನ್ಯ ಪ್ರಶ್ನೆ.** ನಾನು ಕೃಷಿ, ಬೆಳೆಗಳು, ಕೀಟಗಳು, ರಸಗೊಬ್ಬರ, ನೀರಾವರಣ ಅಥವಾ PM-KISAN ನಂತಹ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳಿಗೆ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳಿಗೆ ಮಾತ್ರ ಉತ್ತರ ನೀಡಬಹುದು. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಸುಧಾರಿಸಿ.",
        },
    }
    return invalid_responses.get(language, invalid_responses["English"]).get(
        reason, invalid_responses["English"].get("off-topic", "Invalid query")
    )



def _fetch_json(url: str) -> dict:
    with urlopen(url, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather_forecast(location: str, days: int) -> dict:
    location = location.strip()
    if not location:
        raise ValueError("Location cannot be empty.")

    geo_query = urlencode({"name": location, "count": 1, "language": "en", "format": "json"})
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?{geo_query}"
    geo_data = _fetch_json(geo_url)

    results = geo_data.get("results") or []
    if not results:
        raise ValueError("Location not found. Try nearby city or correct spelling.")

    place = results[0]
    latitude = place["latitude"]
    longitude = place["longitude"]
    place_name = ", ".join(
        part
        for part in [
            place.get("name"),
            place.get("admin1"),
            place.get("country"),
        ]
        if part
    )

    weather_query = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,wind_speed_10m_max",
            "timezone": "auto",
            "forecast_days": max(1, min(days, 16)),
        }
    )
    weather_url = f"https://api.open-meteo.com/v1/forecast?{weather_query}"
    weather_data = _fetch_json(weather_url)
    daily = weather_data.get("daily")
    if not daily:
        raise RuntimeError("Forecast service returned incomplete data. Please try again.")

    dates = daily.get("time", [])
    tmax = daily.get("temperature_2m_max", [])
    tmin = daily.get("temperature_2m_min", [])
    rain_mm = daily.get("precipitation_sum", [])
    rain_prob = daily.get("precipitation_probability_max", [])
    wind = daily.get("wind_speed_10m_max", [])

    horizon = min(days, len(dates), len(tmax), len(tmin), len(rain_mm), len(rain_prob), len(wind))
    forecast = []
    for i in range(horizon):
        forecast.append(
            {
                "date": dates[i],
                "temp_max": tmax[i],
                "temp_min": tmin[i],
                "rain_mm": rain_mm[i],
                "rain_prob": rain_prob[i],
                "wind_kmh": wind[i],
            }
        )

    return {
        "location_name": place_name,
        "lat": latitude,
        "lon": longitude,
        "forecast": forecast,
        "provider": "open-meteo",
    }


def _weather_advice(forecast: list[dict]) -> str:
    if not forecast:
        return "No forecast data available to generate advisory."

    avg_rain_prob = sum(day["rain_prob"] for day in forecast) / len(forecast)
    total_rain = sum(day["rain_mm"] for day in forecast)
    max_temp = max(day["temp_max"] for day in forecast)
    max_wind = max(day["wind_kmh"] for day in forecast)

    tips = []
    if avg_rain_prob >= 60 or total_rain >= 25:
        tips.append("Rain likely: reduce irrigation and avoid fertilizer spray just before expected rain.")
    elif avg_rain_prob <= 25 and total_rain < 5:
        tips.append("Dry window expected: plan irrigation cycles and conserve soil moisture with mulching.")

    if max_temp >= 35:
        tips.append("High temperature risk: irrigate during early morning/evening and monitor crop stress.")

    if max_wind >= 25:
        tips.append("Strong wind expected: avoid pesticide spraying during peak wind hours.")

    if not tips:
        tips.append("Weather looks moderate. Continue regular irrigation and crop monitoring schedule.")

    return "\n\n".join(f"- {tip}" for tip in tips)


_PERISHABLE_CROPS = {"Tomato", "Onion", "Potato", "Cabbage", "Cauliflower"}
_WINDOWS = [0, 7, 14, 21, 30]
_PRICE_DATA_CSV = "data/mandi_prices_daily_2015_2025.csv"


@st.cache_data(ttl=1200, show_spinner=False)
def get_openweather_7day(location: str, api_key: str) -> dict:
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY is missing in .env")

    geo_query = urlencode({"q": location.strip(), "limit": 1, "appid": api_key})
    geo_url = f"https://api.openweathermap.org/geo/1.0/direct?{geo_query}"
    geo_data = _fetch_json(geo_url)
    if not geo_data:
        raise ValueError("Location not found in OpenWeatherMap.")

    place = geo_data[0]
    lat, lon = place["lat"], place["lon"]
    place_name = ", ".join(
        part for part in [place.get("name"), place.get("state"), place.get("country")] if part
    )

    weather_query = urlencode(
        {
            "lat": lat,
            "lon": lon,
            "exclude": "minutely,hourly,alerts",
            "units": "metric",
            "appid": api_key,
        }
    )
    weather_url = f"https://api.openweathermap.org/data/3.0/onecall?{weather_query}"
    weather_data = _fetch_json(weather_url)

    daily = (weather_data or {}).get("daily") or []
    if not daily:
        raise RuntimeError(
            "OpenWeatherMap daily forecast unavailable. Verify One Call API access for your key."
        )

    forecast = []
    for day in daily[:7]:
        dt = datetime.utcfromtimestamp(day["dt"]).strftime("%Y-%m-%d")
        forecast.append(
            {
                "date": dt,
                "temp_max": float(day.get("temp", {}).get("max", 0.0)),
                "temp_min": float(day.get("temp", {}).get("min", 0.0)),
                "rain_mm": float(day.get("rain", 0.0)),
                "rain_prob": float(day.get("pop", 0.0) * 100.0),
                "wind_kmh": float(day.get("wind_speed", 0.0) * 3.6),
            }
        )

    return {"location_name": place_name, "lat": lat, "lon": lon, "forecast": forecast}


@st.cache_data(ttl=21600, show_spinner=False)
def get_nasa_power_baseline(lat: float, lon: float, lookback_days: int = 30) -> dict:
    end_date = datetime.utcnow().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=lookback_days - 1)

    nasa_query = urlencode(
        {
            "parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN",
            "community": "AG",
            "longitude": f"{lon:.4f}",
            "latitude": f"{lat:.4f}",
            "start": start_date.strftime("%Y%m%d"),
            "end": end_date.strftime("%Y%m%d"),
            "format": "JSON",
        }
    )
    nasa_url = f"https://power.larc.nasa.gov/api/temporal/daily/point?{nasa_query}"
    nasa_data = _fetch_json(nasa_url)
    params = ((nasa_data or {}).get("properties") or {}).get("parameter") or {}

    rainfall_series = [
        float(v)
        for v in (params.get("PRECTOTCORR") or {}).values()
        if v is not None and float(v) >= 0
    ]
    tmax_series = [
        float(v) for v in (params.get("T2M_MAX") or {}).values() if v is not None and float(v) > -90
    ]
    tmin_series = [
        float(v) for v in (params.get("T2M_MIN") or {}).values() if v is not None and float(v) > -90
    ]

    if not rainfall_series:
        raise RuntimeError("NASA POWER baseline data unavailable for this location.")

    avg_daily_rain = sum(rainfall_series) / len(rainfall_series)
    return {
        "avg_daily_rain_mm": avg_daily_rain,
        "avg_7d_rain_mm": avg_daily_rain * 7,
        "avg_tmax_c": (sum(tmax_series) / len(tmax_series)) if tmax_series else None,
        "avg_tmin_c": (sum(tmin_series) / len(tmin_series)) if tmin_series else None,
    }


def _fuse_weather_risk(crop: str, owm_forecast: list[dict], nasa_baseline: dict) -> dict:
    total_rain = sum(day["rain_mm"] for day in owm_forecast)
    heavy_rain_days = sum(1 for day in owm_forecast if day["rain_mm"] > 50)
    hot_days = sum(1 for day in owm_forecast if day["temp_max"] >= 35)
    windy_days = sum(1 for day in owm_forecast if day["wind_kmh"] >= 30)

    baseline_7d_rain = nasa_baseline.get("avg_7d_rain_mm", 0.0)
    rain_anomaly = total_rain - baseline_7d_rain

    risk_score = (
        heavy_rain_days * 24
        + hot_days * 8
        + windy_days * 6
        + max(0.0, rain_anomaly) * 0.35
    )
    if crop in _PERISHABLE_CROPS:
        risk_score *= 1.15
    risk_score = max(0.0, min(100.0, risk_score))

    penalty_pct = min(0.40, 0.04 + (risk_score / 100.0) * 0.22)

    alerts = []
    if heavy_rain_days > 0 and crop in _PERISHABLE_CROPS:
        alerts.append(
            "⚠️ Rainfall > 50 mm/24h detected in 7-day outlook. Immediate sell signal for perishables."
        )
    if rain_anomaly > 20:
        alerts.append("⚠️ Rainfall expected above climate baseline (NASA POWER).")
    if hot_days >= 2:
        alerts.append("⚠️ Heat stress window likely for crops in open fields.")
    if windy_days >= 2:
        alerts.append("⚠️ Strong wind conditions may increase handling/logistics loss risk.")

    if not alerts:
        alerts.append("✅ No major weather shock detected in next 7 days.")

    return {
        "score": risk_score,
        "penalty_pct": penalty_pct,
        "alerts": alerts,
        "total_rain_mm_7d": total_rain,
        "baseline_rain_mm_7d": baseline_7d_rain,
        "rain_anomaly_mm": rain_anomaly,
        "heavy_rain_days": heavy_rain_days,
    }


def _base_trend_per_7d(crop: str) -> float:
    trend_map = {
        "Wheat": 0.012,
        "Rice": 0.010,
        "Maize": 0.011,
        "Cotton": 0.014,
        "Soybean": 0.013,
        "Tomato": -0.020,
        "Onion": -0.008,
        "Potato": -0.004,
    }
    return trend_map.get(crop, 0.008)


@st.cache_resource(show_spinner=False)
def _get_price_engine() -> CropPriceForecastingEngine:
    return CropPriceForecastingEngine(_PRICE_DATA_CSV)


@st.cache_data(ttl=21600, show_spinner=False)
def get_engine_price_forecast(crop: str, mandi: str) -> dict:
    engine = _get_price_engine()
    result = engine.forecast_crop_mandi(crop=crop, mandi=mandi)
    return {
        "forecast": result.forecast,
        "lower_ci": result.lower_ci,
        "upper_ci": result.upper_ci,
        "weights": result.model_weights,
        "confidence": result.confidence,
    }


def _get_secret_or_env(key: str, default: str = "") -> str:
    try:
        value = st.secrets.get(key, default)
    except (StreamlitSecretNotFoundError, FileNotFoundError, KeyError):
        value = default
    except Exception:
        value = default

    return str(value).strip() if value else os.getenv(key, default)


def _looks_like_placeholder_api_key(value: str) -> bool:
    if not value:
        return True
    v = value.strip().lower()
    return (
        "your_openweathermap_api_key" in v
        or "replace_me" in v
        or v in {"none", "null", "changeme", "test"}
    )


def _get_hyperlocal_weather(location: str, preferred_api_key: str) -> dict:
    """Try OpenWeatherMap first (if usable), then fallback to Open-Meteo."""
    api_key = (preferred_api_key or "").strip()
    if api_key and not _looks_like_placeholder_api_key(api_key):
        try:
            weather = get_openweather_7day(location, api_key)
            weather["provider"] = "openweathermap"
            return weather
        except HTTPError as http_exc:
            if http_exc.code not in {401, 403}:
                raise
        except Exception:
            pass

    # fallback, no key required
    return get_weather_forecast(location, 7)


def optimize_sell_windows(
    crop: str,
    current_price: float,
    quantity_qtl: float,
    storage_cost_per_day: float,
    transport_cost: float,
    weather_risk: dict,
    model_forecast: Optional[dict] = None,
) -> dict:
    trend_per_7d = _base_trend_per_7d(crop)
    risk_mult_base = 1.25 if crop in _PERISHABLE_CROPS else 1.0
    rows = []

    for window in _WINDOWS:
        trend_factor = 1.0 + trend_per_7d * (window / 7.0)
        expected_price = max(0.0, current_price * trend_factor)

        if model_forecast:
            if window == 0:
                expected_price = float(current_price)
            elif window in model_forecast:
                expected_price = float(model_forecast[window])
            elif window == 21 and 14 in model_forecast and 30 in model_forecast:
                expected_price = float(model_forecast[14] + (model_forecast[30] - model_forecast[14]) * ((21 - 14) / (30 - 14)))

        risk_multiplier = risk_mult_base * (1.0 + window / 30.0)
        weather_penalty_per_qtl = current_price * weather_risk["penalty_pct"] * risk_multiplier
        storage_total = quantity_qtl * storage_cost_per_day * window
        gross_income = expected_price * quantity_qtl
        weather_penalty_total = weather_penalty_per_qtl * quantity_qtl
        net_income = gross_income - storage_total - transport_cost - weather_penalty_total

        rows.append(
            {
                "window_days": window,
                "expected_price": expected_price,
                "gross_income": gross_income,
                "storage_cost": storage_total,
                "transport_cost": transport_cost,
                "weather_penalty": weather_penalty_total,
                "net_income": net_income,
            }
        )

    best = max(rows, key=lambda x: x["net_income"])
    now_row = next(row for row in rows if row["window_days"] == 0)
    impact = best["net_income"] - now_row["net_income"]

    heavy_rain_alert = weather_risk.get("heavy_rain_days", 0) > 0 and crop in _PERISHABLE_CROPS
    if heavy_rain_alert and best["window_days"] > 0:
        recommendation = "Partial Sell"
        rationale = "Severe rain alert for perishables: sell major portion now and hold a smaller portion."
    elif best["window_days"] == 0:
        recommendation = "Sell Now"
        rationale = "Immediate sale gives the highest expected net income after costs and weather penalties."
    elif impact <= max(500.0, 0.03 * max(1.0, now_row["net_income"])):
        recommendation = "Partial Sell"
        rationale = "Future gain is limited versus now; partial sell balances liquidity and upside."
    else:
        recommendation = "Hold"
        rationale = f"Expected net income is highest if sold after {best['window_days']} days."

    confidence = 78.0
    confidence += 8.0 if len(rows) == 5 else 0.0
    confidence += 6.0 if weather_risk["score"] < 40 else -8.0
    confidence += 4.0 if abs(impact) > 1000 else -4.0
    confidence = max(40.0, min(95.0, confidence))

    return {
        "rows": rows,
        "best": best,
        "now": now_row,
        "income_impact": impact,
        "recommendation": recommendation,
        "rationale": rationale,
        "confidence": confidence,
    }

st.set_page_config(page_title="Farmer Advisory Assistant", page_icon="🌾", layout="wide", initial_sidebar_state="collapsed")

# Kannada translations for UI elements - MUST be defined before use
_TRANSLATIONS = {
    "English": {
        # Welcome & Core
        "welcome": "👋 Welcome! Ask me about crops, pests, fertilizer schedules, irrigation, or government schemes like PM-KISAN.",
        "farmer_assistant": "### Farmer Assistant",
        "quick_prompts": "#### Quick prompts",
        "voice_query": "#### Voice query",
        "new_chat": "New chat",
        
        # Quick Questions
        "q1": "How to control fall armyworm in maize?",
        "q2": "Best fertilizer schedule for paddy",
        "q3": "How often should I irrigate tomato in summer?",
        "q4": "PM-KISAN eligibility and required documents",
        
        # Registration
        "farmer_name": "Farmer Name",
        "farmer_name_placeholder": "Enter your full name",
        "place_village": "Place / Village",
        "place_placeholder": "Enter your village, town, or city",
        "mobile_no": "Mobile No.",
        "mobile_placeholder": "10-digit mobile number",
        "language_label": "Preferred Language",
        "registration_incomplete": "Please fill in your name, place, and mobile number.",
        "invalid_mobile": "Please enter a valid 10-digit mobile number.",
        "registration_success": "Registration successful. Welcome, {name}!",
        
        # Query Validation
        "empty_query": "⚠️ Please type a question before sending.",
        "too_short": "⚠️ Your query is too short. Please ask a complete farming-related question.",
        "too_long": "⚠️ Your message is too long. Please shorten your question.",
        "gibberish": "❌ That doesn't look like a valid question. Please ask something about farming, crops, or agriculture.",
        "off_topic": "❌ **Invalid query.** I can only answer questions related to farming, crops, pests, fertilizers, irrigation, or government agricultural schemes like PM-KISAN. Please rephrase your question.",
        
        # Chat Section
        "ask_question": "Ask your farming question...",
        "send_button": "Send",
        "preparing_advisory": "Preparing advisory...",
        "not_registered_chat": "Please register yourself on the Home page first to unlock Farmer Query.",
        "speech_detection_failed": "Could not detect speech. Try again.",
        "speech_error": "Voice transcription failed: {error}",
        "voice_output_failed": "Could not generate voice output: {error}",
        "chat_topbar_title": "AI Farmer Advisory Chat",
        "chat_topbar_subtitle": "Ask about crops, pests, fertilizer, irrigation & govt. schemes",
        "online_status": "Online",
        
        # Weather Section
        "not_registered_weather": "Please register yourself on the Home page first to unlock Weather Prediction.",
        "weather_title": "### Weather Prediction",
        "village_city": "Village / City",
        "village_placeholder": "e.g., Mysuru",
        "forecast_window": "Forecast window",
        "location_required": "Please enter a location.",
        "forecast_loaded": "Forecast loaded for {location} ({days} days).",
        "daily_forecast": "#### Daily forecast",
        "farming_advisory": "#### Farming advisory",
        "weather_error": "Weather prediction failed: {error}",
        
        # Price Section
        "not_registered_price": "Please register yourself on the Home page first to unlock Price Prediction.",
        "price_title": "### Price Prediction",
        "crop_price_forecasting": "#### Crop price forecasting engine",
        "alert_engine": "#### Alert engine",
        "profit_optimizer": "#### Profit optimizer (0/7/14/21/30 days)",
        "recommendation": "#### Recommendation",
        
        # Navigation
        "home": "Home",
        "chat": "Chat",
        "weather": "Weather",
        "price": "Price",
        "profile": "Profile",
        "settings": "Settings",
        "read_aloud": "Read answers aloud",
    },
    "Kannada": {
        # Welcome & Core
        "welcome": "👋 ನಮ್ಮಗೆ ಸ್ವಾಗತ! ಬೆಳೆಗಳು, ಕೀಟಗಳು, ರಸಗೊಬ್ಬರ ಅವಧಿ, ನೀರಾವರಣ ಅಥವಾ PM-KISAN ನಂತಹ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳ ಬಗ್ಗೆ ನನ್ನನ್ನು ಕೇಳಿ.",
        "farmer_assistant": "### ರೈತ ಸಹಾಯಕ",
        "quick_prompts": "#### ತ್ವರಿತ ಪ್ರಶ್ನೆಗಳು",
        "voice_query": "#### ಧ್ವನಿ ಪ್ರಶ್ನೆ",
        "new_chat": "ಹೊಸ ಚಾಟ್",
        
        # Quick Questions
        "q1": "ಮೆಕ್ಕೆಯಲ್ಲಿ ಫಾಲ್ ಆರ್ಮಿವರ್ಮ್ ನಿಯಂತ್ರಣ ಹೇಗೆ?",
        "q2": "ನೆಲೆ ಪದ್ಧತಿಯ ಸೋತಾ ರಸಗೊಬ್ಬರ ವೇಳಾಪಟ್ಟಿ",
        "q3": "ಬೇಸಿಗೆಯಲ್ಲಿ ಟೊಮ್ಯಾಟೋ ಎಷ್ಟು ಬಾರಿ ನೀರಾವರಿಸಬೇಕು?",
        "q4": "PM-KISAN ಯೋಗ್ಯತೆ ಮತ್ತು ಅಗತ್ಯ ದಾಖಲೆಗಳು",
        
        # Registration
        "farmer_name": "ರೈತನ ಹೆಸರು",
        "farmer_name_placeholder": "ನಿಮ್ಮ ಸಂಪೂರ್ಣ ಹೆಸರನ್ನು ನಮೂದಿಸಿ",
        "place_village": "ಸ್ಥಳ / ಗ್ರಾಮ",
        "place_placeholder": "ನಿಮ್ಮ ಗ್ರಾಮ, ಊರು ಅಥವಾ ನಗರವನ್ನು ನಮೂದಿಸಿ",
        "mobile_no": "ಮೊಬೈಲ್ ಸಂಖ್ಯೆ",
        "mobile_placeholder": "10 ಅಂಕಿಯ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ",
        "language_label": "ಆದ್ಯತೆಯ ಭಾಷೆ",
        "registration_incomplete": "ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹೆಸರು, ಸ್ಥಳ ಮತ್ತು ಮೊಬೈಲ್ ಸಂಖ್ಯೆ ಭರ್ತಿ ಮಾಡಿ.",
        "invalid_mobile": "ದಯವಿಟ್ಟು ಸಿದ್ಧ 10 ಅಂಕಿಯ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ ನಮೂದಿಸಿ.",
        "registration_success": "ನೋಂದಣಿ ಯಶಸ್ವಿ. ಸ್ವಾಗತ, {name}!",
        
        # Query Validation
        "empty_query": "⚠️ ಕಳುಹಿಸುವ ಮೊದಲು ದಯವಿಟ್ಟು ಒಂದು ಪ್ರಶ್ನೆ ಟೈಪ್ ಮಾಡಿ.",
        "too_short": "⚠️ ನಿಮ್ಮ ಪ್ರಶ್ನೆ ತುಂಬಾ ಚಿಕ್ಕದಾಗಿದೆ. ದಯವಿಟ್ಟು ಸಂಪೂರ್ಣ ಕೃಷಿ-ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ.",
        "too_long": "⚠️ ನಿಮ್ಮ ಸಂದೇಶ ತುಂಬಾ ಉದ್ದವಾಗಿದೆ. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಸಂಕ್ಷಿಪ್ತಗೊಳಿಸಿ.",
        "gibberish": "❌ ಇದು ಸಿಂಧುವಾಗಿ ತೋರುತ್ತಿಲ್ಲ. ದಯವಿಟ್ಟು ಕೃಷಿ, ಬೆಳೆಗಳು ಅಥವಾ ಕೃಷಿಯ ಬಗ್ಗೆ ಏನನ್ನಾದರೂ ಕೇಳಿ.",
        "off_topic": "❌ **ಅಮಾನ್ಯ ಪ್ರಶ್ನೆ.** ನಾನು ಕೃಷಿ, ಬೆಳೆಗಳು, ಕೀಟಗಳು, ರಸಗೊಬ್ಬರ, ನೀರಾವರಣ ಅಥವಾ PM-KISAN ನಂತಹ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳಿಗೆ ಸಂಬಂಧಿತ ಪ್ರಶ್ನೆಗಳಿಗೆ ಮಾತ್ರ ಉತ್ತರ ನೀಡಬಹುದು. ದಯವಿಟ್ಟು ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಸುಧಾರಿಸಿ.",
        
        # Chat Section
        "ask_question": "ನಿಮ್ಮ ಕೃಷಿ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ...",
        "send_button": "ಕಳುಹಿಸಿ",
        "preparing_advisory": "ಸಲಹೆ ತಯಾರಿಸಿ...",
        "not_registered_chat": "ಫಾರ್ಮರ್ ಕೇರಿ ಅನ್‌ಲಾಕ್ ಮಾಡಲು ದಯವಿಟ್ಟು ಹೋಮ್ ಪುಟದಲ್ಲಿ ನೋಂದಾಯನ ಮಾಡಿ.",
        "speech_detection_failed": "ಧ್ವನಿ ಶೋಧಿತವಾಗಿಲ್ಲ. ಪುನಃ ಪ್ರಯತ್ನಿಸಿ.",
        "speech_error": "ವಾಯ್ಸ್ ಟ್ರಾನ್ಸ್ಕ್ರಿಪ್ಷನ ವಿಫಲ: {error}",
        "voice_output_failed": "ವಾಯ್ಸ್ ಔಟ್‌ಪುಟ್ ತಯಾರಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ: {error}",
        "chat_topbar_title": "AI ರೈತರ ಸಲಹಾ ಚಾಟ್",
        "chat_topbar_subtitle": "ಬೆಳೆ, ಕೀಟ, ರಸಗೊಬ್ಬರ, ನೀರಾವರಿ ಮತ್ತು ಸರ್ಕಾರದ ಯೋಜನೆಗಳ ಬಗ್ಗೆ ಕೇಳಿ",
        "online_status": "ಆನ್‌ಲೈನ್",
        
        # Weather Section
        "not_registered_weather": "ವಾತಾವರಣ ಮುನ್ನೆಚ್ಚರಣೆ ಅನ್‌ಲಾಕ್ ಮಾಡಲು ದಯವಿಟ್ಟು ಹೋಮ್ ಪುಟದಲ್ಲಿ ನೋಂದಾಯನ ಮಾಡಿ.",
        "weather_title": "### ವಾತಾವರಣ ಮುನ್ನೆಚ್ಚರಣೆ",
        "village_city": "ಗ್ರಾಮ / ನಗರ",
        "village_placeholder": "ಉದಾ: ಮೈಸೂರು",
        "forecast_window": "ಮುನ್ನೆಚ್ಚರಣೆ ಅವಧಿ",
        "location_required": "ದಯವಿಟ್ಟು ಸ್ಥಳವನ್ನು ನಮೂದಿಸಿ.",
        "forecast_loaded": "{location} ಗಾಗಿ ಮುನ್ನೆಚ್ಚರಣೆ ಲೋಡ್ ಆಗಿದೆ ({days} ದಿನಗಳು).",
        "daily_forecast": "#### ದೈನಿಕ ಮುನ್ನೆಚ್ಚರಣೆ",
        "farming_advisory": "#### ಕೃಷಿ ಸಲಹೆ",
        "weather_error": "ವಾತಾವರಣ ಮುನ್ನೆಚ್ಚರಣೆ ವಿಫಲ: {error}",
        
        # Price Section
        "not_registered_price": "ಬೆಲೆ ಮುನ್ನೆಚ್ಚರಣೆ ಅನ್‌ಲಾಕ್ ಮಾಡಲು ದಯವಿಟ್ಟು ಹೋಮ್ ಪುಟದಲ್ಲಿ ನೋಂದಾಯನ ಮಾಡಿ.",
        "price_title": "### ಬೆಲೆ ಮುನ್ನೆಚ್ಚರಣೆ",
        "crop_price_forecasting": "#### ಬೆಳೆಯ ಬೆಲೆ ಮುನ್ನೆಚ್ಚರಣೆ ಇಂಜಿನ್",
        "alert_engine": "#### ಎಚ್ಚರಿಕೆ ಇಂಜಿನ್",
        "profit_optimizer": "#### ಲಾಭ ಆಪ್ಟಿಮೈজರ್ (0/7/14/21/30 ದಿನಗಳು)",
        "recommendation": "#### ಸಿಫಾರಿಶ",
        
        # Navigation
        "home": "ನೆಲೆ",
        "chat": "ಚಾಟ್",
        "weather": "ವಾತಾವರಣ",
        "price": "ಬೆಲೆ",
        "profile": "ಪ್ರೊಫೈಲ್",
        "settings": "ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
        "read_aloud": "ಉತ್ತರಗಳನ್ನು ಜೋರಾಗಿ ಓದಿ",
    },
}

def _get_translation(key: str, lang: str = "English") -> str:
    """Get translated text for a given key and language."""
    return _TRANSLATIONS.get(lang, {}).get(key, _TRANSLATIONS["English"].get(key, key))
    return _TRANSLATIONS.get(lang, {}).get(key, _TRANSLATIONS["English"].get(key, key))


def _load_logo_data_uri() -> str:
    """Return data URI for the project logo image, or empty string if missing."""
    logo_path = os.path.join("pics", "Screenshot 2026-04-21 143316.png")
    try:
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
    except Exception:
        return ""


def _seed_welcome_message():
    lang = st.session_state.response_language if "response_language" in st.session_state else "English"
    return [
        {
            "role": "assistant",
            "content": _get_translation("welcome", lang),
            "audio": None,
        }
    ]


if "messages" not in st.session_state:
    st.session_state.messages = _seed_welcome_message()

if "last_mic_hash" not in st.session_state:
    st.session_state.last_mic_hash = None

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "pending_display_text" not in st.session_state:
    st.session_state.pending_display_text = None

if "enable_tts" not in st.session_state:
    st.session_state.enable_tts = False

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Light"

if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Home"

if "menu_open" not in st.session_state:
    st.session_state.menu_open = False
if "farmer_registered" not in st.session_state:
    st.session_state.farmer_registered = False

if "farmer_profile" not in st.session_state:
    st.session_state.farmer_profile = {
        "name": "",
        "place": "",
        "mobile": "",
        "language": "English",
    }

if "response_language" not in st.session_state:
    st.session_state.response_language = "English"

_THEMES = {
    "Light": {
        "app_bg": "#000000",
        "topbar_bg": "#000000",
        "topbar_text": "#ffffff",
        "topbar_subtext": "#ffffff",
        "chat_bg": "#ffffff",
        "composer_bg": "#ffffff",
        "composer_text": "#000000",
        "composer_placeholder": "#666666",
        "composer_border": "#ffffff",
    },
    "Dark": {
        "app_bg": "#000000",
        "topbar_bg": "#000000",
        "topbar_text": "#ffffff",
        "topbar_subtext": "#ffffff",
        "chat_bg": "#121212",
        "composer_bg": "#1e1e1e",
        "composer_text": "#ffffff",
        "composer_placeholder": "#9aa0a6",
        "composer_border": "#2a2a2a",
    },
}

current_theme = _THEMES["Light"]
css = """
<style>
    :root {
        --chat-input-width: min(1100px, calc(100vw - 1.2rem));
        --composer-bottom: 0.65rem;
        --mic-size: 2.2rem;
        --green-600: #ffffff;
        --green-700: #ffffff;
        --green-50: #ffffff;
        --radius-pill: 999px;
        --radius-card: 16px;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
        block-size: 0 !important;
    }

    [data-testid="stSidebarCollapsedControl"],
    button[title="Open sidebar"],
    button[title="Close sidebar"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        z-index: 1200 !important;
        position: fixed !important;
        inset-block-start: 0.55rem !important;
        inset-inline-start: auto !important;
        inset-inline-end: 0.55rem !important;
        inline-size: 2.4rem !important;
        block-size: 2.4rem !important;
        min-inline-size: 44px !important;
        min-block-size: 44px !important;
        border-radius: 8px !important;
        background: #000000 !important;
        border: 1px solid #ffffff !important;
        color: #ffffff !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        opacity: 1 !important;
        visibility: visible !important;
        pointer-events: auto !important;
    }

    [data-testid="stSidebarCollapsedControl"] button,
    [data-testid="stSidebarCollapsedControl"] svg,
    button[title="Open sidebar"] svg,
    button[title="Close sidebar"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        opacity: 1 !important;
    }

    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"] {
        block-size: 100%;
    }
    [data-testid="stAppViewContainer"] { background: __APP_BG__; }

    .main .block-container {
        max-inline-size: 1120px;
        padding-block-start: 5.2rem;
        padding-block-end: 7rem;
        padding-inline: 1.2rem;
    }

    .chat-topbar {
        position: fixed;
        inset-block-start: 0;
        inset-inline-start: 2.8rem;
        inset-inline-end: 0;
        z-index: 999;
        background: __TOPBAR_BG__;
        padding: 0.82rem 1.5rem;
        color: __TOPBAR_TEXT__;
        box-shadow: 0 6px 22px rgba(255, 255, 255, 0.34);
        display: flex;
        align-items: center;
        gap: 0.85rem;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        transition: inset-inline-start 0.3s ease, padding 0.25s ease;
    }

    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar {
        inset-inline-start: var(--sidebar-width, 336px);
        padding: 0.65rem 1.1rem;
    }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar h2 { font-size: 0.95rem; }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar .topbar-icon { font-size: 1.45rem; }
    [data-testid="stApp"]:has([data-testid="stSidebar"][aria-expanded="true"]) .chat-topbar p { font-size: 0.76rem; }

    .chat-topbar .topbar-icon { font-size: 2rem; line-height: 1; flex-shrink: 0; }
    .chat-topbar .topbar-text { flex: 1; min-inline-size: 0; }
    .chat-topbar .topbar-logo {
        max-block-size: 68px;
        inline-size: auto;
        display: block;
        object-fit: contain;
    }
    .chat-topbar h2 {
        margin: 0;
        font-size: 1.18rem;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .chat-topbar p {
        margin: 0.15rem 0 0 0;
        color: __TOPBAR_SUBTEXT__;
        font-size: 0.84rem;
        opacity: 0.92;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .topbar-status {
        display: flex;
        align-items: center;
        gap: 0.38rem;
        font-size: 0.78rem;
        color: __TOPBAR_SUBTEXT__;
        background: rgba(255,255,255,0.12);
        border-radius: var(--radius-pill);
        padding: 0.28rem 0.72rem;
        font-weight: 500;
        flex-shrink: 0;
        white-space: nowrap;
    }
    .topbar-status::before {
        content: "";
        display: inline-block;
        inline-size: 0.52rem;
        block-size: 0.52rem;
        border-radius: 50%;
        background: #fde047;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot { 0%,100% { opacity:1; } 50% { opacity:0.4; } }

    [data-testid="stChatMessage"] {
        padding: 0.5rem 0.6rem !important;
        border-radius: var(--radius-card) !important;
        margin-block-end: 0.55rem !important;
        border: none !important;
        transition: background 0.15s;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: #ffffff !important;
        border: 1px solid #ffffff !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: #000000 !important;
        border: none !important;
        box-shadow: 0 3px 10px rgba(255,255,255,0.11);
    }
    [data-testid="stChatMessageAvatarUser"] {
        background: #ffffff !important;
        border-radius: 50% !important;
        color: #fff !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background: #ffffff !important;
        border-radius: 50% !important;
        color: #fff !important;
    }
    [data-testid="stChatMessage"] p { font-size: 0.97rem; line-height: 1.65; color: #111827; }

    div[data-testid="stBottomBlockContainer"] {
        padding-inline: 0.55rem;
        padding-block-end: var(--composer-bottom);
        background: #000000;
    }

    div[data-testid="stChatInput"] {
        inline-size: 100%;
        max-inline-size: var(--chat-input-width);
        margin-inline: auto;
        padding-inline-start: 0.6rem;
    }
    div[data-testid="stChatInput"] > div {
        border-radius: 30px !important;
        border: 1.5px solid #ffffff !important;
        background: #000000 !important;
        box-shadow: 0 4px 24px rgba(255,255,255,0.18), 0 1px 4px rgba(0,0,0,0.07) !important;
        padding-block: 0.35rem;
        transition: border-color 0.18s, box-shadow 0.18s;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.18), 0 4px 20px rgba(255,255,255,0.18) !important;
    }
    div[data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        font-size: 1rem !important;
        padding-inline-start: 0.4rem !important;
        padding-block: 0.28rem !important;
        background: transparent !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder { color: #9ca3af !important; opacity: 1; }
    div[data-testid="stChatInput"] button {
        border-radius: var(--radius-pill) !important;
        inline-size: 2.6rem !important;
        block-size: 2.6rem !important;
        min-inline-size: 44px !important;
        min-block-size: 44px !important;
        background: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 10px rgba(255,255,255,0.34) !important;
        transition: transform 0.12s, box-shadow 0.12s;
    }
    div[data-testid="stChatInput"] button:hover { transform: scale(1.06); }
    div[data-testid="stChatInput"] button svg { display: none !important; }
    div[data-testid="stChatInput"] button::before { content: "➤"; color: #fff; font-size: 1rem; }

    [data-testid="stSidebar"] {
        background: #000000 !important;
        border-inline-end: 1px solid #ffffff !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        font-weight: 700;
        font-size: 1.05rem;
        color: #ffffff;
        border-block-end: 2px solid #ffffff;
        padding-block-end: 0.35rem;
        margin-block-end: 0.6rem;
    }
    [data-testid="stSidebar"] .stMarkdown h4 {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280;
        margin-block-end: 0.4rem;
        margin-block-start: 1rem;
    }
    [data-testid="stSidebar"] button[kind="secondary"] {
        border-radius: 10px !important;
        border: 1px solid #ffffff !important;
        background: #000000 !important;
        color: #ffffff !important;
        font-size: 0.84rem !important;
        font-weight: 500 !important;
        text-align: start !important;
        transition: background 0.14s, border-color 0.14s;
        padding-block: 0.6rem !important;
        min-block-size: 44px !important;
    }
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #ffffff !important;
        border-color: #ffffff !important;
    }
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:last-of-type {
        border-radius: 10px !important;
        background: #ffffff !important;
        color: #fff !important;
        border: none !important;
        font-weight: 600 !important;
        min-block-size: 44px !important;
    }

    .sidebar-mic {
        background: #000000;
        border: 1px solid #ffffff;
        border-radius: 12px;
        padding: 0.65rem 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sidebar-mic div[data-testid="stAudioInput"] label { display: none !important; }
    .sidebar-mic button {
        inline-size: var(--mic-size) !important;
        block-size: var(--mic-size) !important;
        min-inline-size: 44px !important;
        min-block-size: 44px !important;
        padding: 0 !important;
        border-radius: var(--radius-pill) !important;
        border: 1px solid #ffffff !important;
        background: #ffffff !important;
    }
    .sidebar-mic button svg { display: none !important; }
    .sidebar-mic button::before { content: "🎤"; font-size: 1.05rem; line-height: 1; }

    .stSpinner > div { border-block-start-color: var(--green-600) !important; }

    .page-card {
        background: #000000;
        border: 1px solid #ffffff;
        border-radius: 14px;
        padding: 1rem;
        box-shadow: 0 6px 18px rgba(255,255,255,0.12);
        margin-block-start: 0.35rem;
    }

    .home-shell {
        position: relative;
        display: grid;
        gap: 0.85rem;
    }

    .home-hero {
        position: relative;
        overflow: hidden;
        background: #000000;
        border: 1px solid #ffffff;
        border-radius: 20px;
        padding: 1.25rem;
        box-shadow: 0 12px 26px rgba(255, 255, 255, 0.18);
        margin-block-end: 0.15rem;
    }

    .home-hero::before,
    .home-hero::after {
        content: "";
        position: absolute;
        border-radius: 999px;
        pointer-events: none;
    }

    .home-hero::before {
        inline-size: 190px;
        block-size: 190px;
        inset-block-start: -65px;
        inset-inline-end: -45px;
        background: radial-gradient(circle, rgba(255,255,255,0.25), rgba(255,255,255,0));
    }

    .home-hero::after {
        inline-size: 170px;
        block-size: 170px;
        inset-block-end: -70px;
        inset-inline-start: -40px;
        background: radial-gradient(circle, rgba(255,255,255,0.24), rgba(255,255,255,0));
    }

    .hero-grid {
        position: relative;
        z-index: 1;
        display: grid;
        grid-template-columns: 1.45fr 1fr;
        gap: 0.9rem;
        align-items: center;
    }

    .home-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.26rem 0.7rem;
        background: #000000;
        border: 1px solid #ffffff;
        color: #ffffff;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .home-hero h1 {
        margin: 0.65rem 0 0.45rem;
        color: #ffffff;
        font-size: clamp(1.4rem, 2.35vw, 2.2rem);
        line-height: 1.2;
    }

    .home-sub {
        margin: 0;
        color: #ffffff;
        font-size: 1rem;
        line-height: 1.6;
    }

    .hero-right {
        display: grid;
        gap: 0.52rem;
    }

    .hero-mini-card {
        border-radius: 12px;
        border: 1px solid #ffffff;
        background: #000000;
        padding: 0.62rem 0.7rem;
        box-shadow: 0 5px 15px rgba(255,255,255,0.12);
    }

    .hero-mini-card strong {
        display: block;
        color: #ffffff;
        font-size: 0.86rem;
        margin-block-end: 0.2rem;
    }

    .hero-mini-card span {
        font-size: 0.8rem;
        color: #ffffff;
        line-height: 1.4;
    }

    .home-stat-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.5rem;
    }

    .home-stat {
        border-radius: 12px;
        border: 1px solid #ffffff;
        background: #000000;
        text-align: center;
        padding: 0.45rem 0.4rem;
    }

    .home-stat strong {
        display: block;
        color: #ffffff;
        font-size: 0.92rem;
    }

    .home-stat small {
        color: #ffffff;
        font-size: 0.72rem;
    }

    .home-section {
        background: #000000;
        border: 1px solid #ffffff;
        border-radius: 16px;
        padding: 1.05rem;
        margin-block: 0.75rem;
        box-shadow: 0 8px 18px rgba(255, 255, 255, 0.11);
    }

    .home-section h3 {
        margin: 0;
        color: #ffffff;
        font-size: 1.08rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .home-section p,
    .home-section li {
        color: #ffffff;
        line-height: 1.6;
        margin-block: 0.45rem;
    }

    .feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin-block-start: 0.7rem;
    }

    .feature-card {
        border-radius: 14px;
        border: 1px solid #ffffff;
        background: #000000;
        padding: 0.78rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .feature-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(255,255,255,0.14);
    }

    .feature-card strong {
        color: #ffffff;
        display: block;
        margin-block-end: 0.25rem;
    }

    .cta-banner {
        border-radius: 16px;
        border: 1px solid #ffffff;
        background: #ffffff;
        color: #000000;
        padding: 1rem;
        margin-block-start: 0.7rem;
        box-shadow: 0 10px 24px rgba(255, 255, 255, 0.25);
    }

    .footer-tagline {
        text-align: center;
        font-weight: 700;
        color: #ffffff;
        margin: 1rem 0 0.2rem;
        font-size: 0.97rem;
    }

    .footer-rotator {
        margin: 0.9rem 0 0.25rem;
        border-radius: 14px;
        border: 1px solid #ffffff;
        background: #000000;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(255,255,255,0.14);
    }

    .footer-track {
        display: inline-flex;
        gap: 0.7rem;
        align-items: center;
        padding: 0.62rem 0.7rem;
        min-inline-size: max-content;
        animation: footer-slide 24s linear infinite;
    }

    .footer-pill {
        border-radius: 999px;
        border: 1px solid #ffffff;
        background: #000000;
        color: #ffffff;
        font-size: 0.84rem;
        font-weight: 600;
        padding: 0.32rem 0.7rem;
        white-space: nowrap;
    }

    @keyframes footer-slide {
        from {
            transform: translateX(0);
        }
        to {
            transform: translateX(-50%);
        }
    }

    .menu-current {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        border: 1px solid #ffffff;
        background: #000000;
        color: #ffffff;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        margin-block-end: 0.55rem;
    }

    .menu-hint {
        font-size: 0.76rem;
        color: #ffffff;
        margin: 0.35rem 0 0.55rem;
    }

    [data-testid="stForm"] {
        border: 1px solid #ffffff;
        border-radius: 12px;
        padding: 0.8rem;
        background: #000000;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        min-block-size: 44px !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stButton"] button {
        min-block-size: 44px !important;
    }

    @media (max-inline-size: 900px) {
        .main .block-container {
            padding-block-start: 4.8rem;
            padding-block-end: 7rem;
            padding-inline: 0.9rem;
        }
        .chat-topbar h2 { font-size: 1.05rem; }
        .chat-topbar p  { font-size: 0.78rem; }

        .hero-grid {
            grid-template-columns: 1fr;
        }

        .feature-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-inline-size: 600px) {
        :root {
            --chat-input-width: calc(100vw - 0.5rem);
            --composer-bottom: env(safe-area-inset-bottom, 0.5rem);
            --mic-size: 2.75rem;
        }
        .chat-topbar {
            inset-inline-start: 0 !important;
            padding: 0.6rem 3.4rem 0.6rem 0.75rem;
            gap: 0.5rem;
        }
        .chat-topbar .topbar-icon { font-size: 1.5rem; }
        .chat-topbar .topbar-logo { max-block-size: 46px; }
        .chat-topbar h2 { font-size: 0.92rem; }
        .chat-topbar p  { display: none; }
        .topbar-status  { display: none; }
        .main .block-container {
            max-inline-size: 100%;
            padding-block-start: 4rem;
            padding-block-end: 5.5rem;
            padding-inline: 0.4rem;
        }
        [data-testid="stChatMessage"] {
            padding: 0.45rem 0.5rem !important;
            border-radius: 12px !important;
            margin-block-end: 0.4rem !important;
        }
        [data-testid="stChatMessage"] p { font-size: 0.93rem; line-height: 1.6; }
        div[data-testid="stBottomBlockContainer"] {
            padding-inline: 0.3rem;
            padding-block-end: max(var(--composer-bottom), 0.4rem);
        }
        div[data-testid="stChatInput"] { padding-inline-start: 0.2rem; max-inline-size: 100%; }
        div[data-testid="stChatInput"] textarea { font-size: 0.97rem !important; }
        [data-testid="stSidebar"] {
            min-inline-size: 100vw !important;
            max-inline-size: 100vw !important;
        }
        [data-testid="stSidebar"] button[kind="secondary"] {
            font-size: 0.9rem !important;
            padding-block: 0.7rem !important;
        }

        .page-card {
            padding: 0.75rem;
            border-radius: 12px;
        }

        .feature-grid {
            grid-template-columns: 1fr;
        }

        .home-stat-row {
            grid-template-columns: 1fr;
        }

        .footer-pill {
            font-size: 0.78rem;
            padding: 0.3rem 0.58rem;
        }

        [data-testid="stForm"] {
            padding: 0.65rem;
        }

        div[data-testid="stTextInput"] input,
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        div[data-testid="stFormSubmitButton"] button,
        div[data-testid="stButton"] button {
            font-size: 0.95rem !important;
        }
    }

    @media (max-inline-size: 380px) {
        .chat-topbar h2 { font-size: 0.82rem; }
        [data-testid="stChatMessage"] p { font-size: 0.88rem; }
        div[data-testid="stChatInput"] textarea { font-size: 0.92rem !important; }
    }
</style>
"""

css = css.replace("__APP_BG__", current_theme["app_bg"])
css = css.replace("__TOPBAR_BG__", current_theme["topbar_bg"])
css = css.replace("__TOPBAR_TEXT__", current_theme["topbar_text"])
css = css.replace("__TOPBAR_SUBTEXT__", current_theme["topbar_subtext"])

st.markdown(css, unsafe_allow_html=True)

bw_css_override = """
<style>
    :root {
        --green-600: #ffffff !important;
        --green-700: #ffffff !important;
        --green-50: #000000 !important;
    }

    html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], .main {
        background: #000000 !important;
        color: #ffffff !important;
    }

    .chat-topbar,
    .page-card,
    .home-hero,
    .home-section,
    .feature-card,
    .hero-mini-card,
    .home-stat,
    .cta-banner,
    .footer-rotator,
    .footer-pill,
    .menu-current,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] button[kind="secondary"],
    [data-testid="stSidebar"] button[kind="primary"],
    .sidebar-mic,
    .sidebar-mic button,
    div[data-testid="stBottomBlockContainer"],
    div[data-testid="stChatInput"] > div,
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]),
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]),
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stForm"],
    [data-testid="stSidebarCollapsedControl"],
    button[title="Open sidebar"],
    button[title="Close sidebar"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        background: #000000 !important;
        background-image: none !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebarCollapsedControl"],
    button[title="Open sidebar"],
    button[title="Close sidebar"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        background: #000000 !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
    }

    [data-testid="stSidebarCollapsedControl"] svg,
    button[title="Open sidebar"] svg,
    button[title="Close sidebar"] svg,
    button[aria-label="Open sidebar"] svg,
    button[aria-label="Close sidebar"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        border: none !important;
    }

    .chat-topbar p,
    .topbar-status,
    .home-sub,
    .menu-hint,
    .home-section p,
    .home-section li,
    .hero-mini-card span,
    .home-stat small,
    .footer-tagline,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stChatMessage"] p {
        color: #ffffff !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background: #000000 !important;
        color: #ffffff !important;
        border-color: #ffffff !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #ffffff !important;
        opacity: 0.7 !important;
    }

    div[data-testid="stChatInput"] button,
    div[data-testid="stFormSubmitButton"] button,
    div[data-testid="stButton"] button {
        background: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        box-shadow: none !important;
    }

    /* Hamburger menu button styling */
    div[data-testid="stButton"] button[kind="primary"] {
        background: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
        font-size: 1.5rem !important;
    }

    div[data-testid="stChatInput"] button:hover,
    div[data-testid="stFormSubmitButton"] button:hover,
    div[data-testid="stButton"] button:hover,
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #ffffff !important;
        color: #000000 !important;
        border-color: #ffffff !important;
        transform: none !important;
    }

    .topbar-status::before,
    .home-hero::before,
    .home-hero::after {
        display: none !important;
        content: none !important;
    }

    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }

    .footer-track {
        animation: footer-slide 24s linear infinite !important;
    }

    /* Override Streamlit alert elements (st.info, st.success, st.warning, st.error) */
    [data-testid="stAlert"],
    [data-testid="stCallout"],
    .stAlert,
    .stCallout,
    .element-container [data-testid="stAlert"],
    .element-container .stAlert {
        background: #000000 !important;
        border: 1px solid #ffffff !important;
        color: #ffffff !important;
    }

    [data-testid="stAlert"] svg,
    [data-testid="stCallout"] svg,
    .stAlert svg,
    .stCallout svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }

    [data-testid="stAlert"] > div,
    [data-testid="stCallout"] > div,
    .stAlert > div,
    .stCallout > div {
        color: #ffffff !important;
    }

    [data-testid="stAlert"] p,
    [data-testid="stCallout"] p,
    .stAlert p,
    .stCallout p {
        color: #ffffff !important;
    }

    /* Override Streamlit metric elements */
    [data-testid="metric-container"],
    .stMetric,
    .element-container .stMetric {
        background: #000000 !important;
        border: 1px solid #ffffff !important;
    }

    [data-testid="metric-container"] > div,
    .stMetric > div {
        color: #ffffff !important;
    }

    /* Override Streamlit dataframe styling */
    [data-testid="stDataFrame"],
    .stDataframe,
    .stDataFrame table {
        background: #000000 !important;
        color: #ffffff !important;
    }

    [data-testid="stDataFrame"] thead th,
    .stDataframe table thead th,
    .stDataFrame table thead th {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #ffffff !important;
    }

    [data-testid="stDataFrame"] tbody td,
    .stDataframe table tbody td,
    .stDataFrame table tbody td {
        border: 1px solid #ffffff !important;
        color: #ffffff !important;
    }

    /* Override caption and section headers */
    [data-testid="stCaption"],
    .stCaption,
    .element-container .stCaption {
        color: #ffffff !important;
    }

    /* Override heading colors */
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4,
    [data-testid="stMarkdownContainer"] h5,
    [data-testid="stMarkdownContainer"] h6 {
        color: #ffffff !important;
    }

    /* Override all text in markdown containers */
    [data-testid="stMarkdownContainer"] {
        color: #ffffff !important;
    }

    [data-testid="stMarkdownContainer"] a {
        color: #ffffff !important;
    }

    /* Override form labels */
    [data-testid="stSelectbox"] label,
    [data-testid="stTextInput"] label,
    [data-testid="stForm"] label,
    .stLabel {
        color: #ffffff !important;
    }

    /* Override select box options */
    [data-baseweb="select"] [role="option"] {
        background: #000000 !important;
        color: #ffffff !important;
    }

    /* Override expander styling */
    [data-testid="stExpander"] button {
        background: #000000 !important;
        color: #ffffff !important;
        border: 1px solid #ffffff !important;
    }

    [data-testid="stExpander"] svg {
        color: #ffffff !important;
    }
</style>
"""

st.markdown(bw_css_override, unsafe_allow_html=True)

mic_audio = None

with st.sidebar:
    st.markdown(_get_translation("farmer_assistant", st.session_state.response_language))
    menu_lang = st.session_state.response_language

    if st.button("☰", key="hamburger_toggle", use_container_width=True, type="primary"):
        st.session_state.menu_open = not st.session_state.menu_open
        st.rerun()

    page_options = [
        ("Home", _get_translation("home", menu_lang)),
        ("Farmer Query", f"🌾 {_get_translation('chat', menu_lang)}"),
        ("Weather Prediction", f"⛅ {_get_translation('weather', menu_lang)}"),
        ("Price Prediction", f"📈 {_get_translation('price', menu_lang)}"),
    ]

    if st.session_state.menu_open:
        for page_value, label in page_options:
            active = st.session_state.selected_page == page_value
            if st.button(
                label,
                key=f"nav_{page_value}",
                use_container_width=True,
                type="primary" if active else "secondary",
            ):
                st.session_state.selected_page = page_value
                st.session_state.menu_open = False
                st.rerun()

    selected_page = st.session_state.selected_page

    if selected_page == "Farmer Query":
        st.session_state.enable_tts = st.toggle(
            _get_translation("read_aloud", st.session_state.response_language),
            value=st.session_state.enable_tts,
        )

        st.markdown(_get_translation("quick_prompts", st.session_state.response_language))
        quick_questions = [
            (_get_translation("q1", st.session_state.response_language), "How to control fall armyworm in maize?"),
            (_get_translation("q2", st.session_state.response_language), "Best fertilizer schedule for paddy"),
            (_get_translation("q3", st.session_state.response_language), "How often should I irrigate tomato in summer?"),
            (_get_translation("q4", st.session_state.response_language), "PM-KISAN eligibility and required documents"),
        ]

        for idx, (display_text, query_text) in enumerate(quick_questions, start=1):
            if st.button(display_text, key=f"quick_{idx}", use_container_width=True):
                st.session_state.pending_prompt = query_text
                st.session_state.pending_display_text = display_text
                st.rerun()

        st.markdown(_get_translation("voice_query", st.session_state.response_language))
        st.markdown("<div class='sidebar-mic'>", unsafe_allow_html=True)
        mic_audio = st.audio_input(" ", key="mic_input")
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button(_get_translation("new_chat", st.session_state.response_language), use_container_width=True):
            st.session_state.messages = _seed_welcome_message()
            st.session_state.pending_prompt = None
            st.session_state.last_mic_hash = None
            st.rerun()

if selected_page == "Home":
    is_kn = st.session_state.response_language == "Kannada"
    home_topbar_logo = _load_logo_data_uri()

    if is_kn:
        home_topbar_icon = "ಮುಖಪುಟ"
        home_topbar_title = "ರೈತರ ಸಲಹಾ ಮುಖಪುಟ"
        online_status = "ಆನ್‌ಲೈನ್"
        home_badge = "ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ವೇದಿಕೆ"
        home_hero_title = "ಸ್ಮಾರ್ಟ್ ನಿರ್ಧಾರಗಳಿಂದ ರೈತರ ಸಬಲೀಕರಣ"
        home_subtitle = "ಬೆಳೆ ಸಲಹೆ, ಹವಾಮಾನ ನವೀಕರಣ, ರೋಗ ಎಚ್ಚರಿಕೆ ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳನ್ನು ನಿಮ್ಮ ಸ್ಥಳೀಯ ಭಾಷೆಯಲ್ಲಿ ಒಂದೇ ಜಾಗದಲ್ಲಿ ಪಡೆಯಿರಿ."
        signal_title = "ಲೈವ್ ಕೃಷಿ ಸೂಚನೆಗಳು"
        signal_desc = "ಹವಾಮಾನ ಎಚ್ಚರಿಕೆ, ಬೆಳೆ ಬೆಂಬಲ ಮತ್ತು ಮಂಡಿ ಧೋರಣೆಗಳು ಒಂದೇ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್‌ನಲ್ಲಿ."
        local_lang_title = "ಸ್ಥಳೀಯ ಭಾಷೆ ಸಿದ್ಧ"
        local_lang_desc = "ರೈತರಿಗೆ ಸುಲಭವಾಗಿ ಅರ್ಥವಾಗುವ ಮತ್ತು ತಕ್ಷಣ ಅನುಸರಿಸಬಹುದಾದ ಮಾರ್ಗದರ್ಶನ."
        stat_1 = "AI ಸಲಹೆ"
        stat_2 = "ಹವಾಮಾನ ಸೂಚನೆಗಳು"
        stat_3 = "ಮಾರುಕಟ್ಟೆ ಸಮಯ"
        register_label = "ರೈತರ ನೋಂದಣಿ"
        registered_farmer_label = "ನೋಂದಾಯಿತ ರೈತ"
        name_label = "ಹೆಸರು"
        place_label = "ಸ್ಥಳ"
        mobile_label = "ಮೊಬೈಲ್"
        language_label = "ಭಾಷೆ"
        registered_note = "ಈಗ ನೀವು ರೈತ ಪ್ರಶ್ನೆ, ಹವಾಮಾನ ಮುನ್ನೋಟ ಮತ್ತು ಬೆಲೆ ಮುನ್ನೋಟ ಆಯ್ಕೆಗಳನ್ನು ಬಳಸಬಹುದು."
        about_section = "ಮಾಹಿತಿ ವಿಭಾಗ"
        about_title = "ರೈತ ಸಲಹಾ ವ್ಯವಸ್ಥೆ ಎಂದರೇನು?"
        about_desc = "ನಮ್ಮ ರೈತ ಸಲಹಾ ವ್ಯವಸ್ಥೆ ಕೃಷಿಕರಿಗೆ ತಿಳಿದ ನಿರ್ಧಾರಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಲು ಸಹಾಯ ಮಾಡುವ AI ಆಧಾರಿತ ವೇದಿಕೆಯಾಗಿದೆ. ಇದು ಬೆಳೆ ಆಯ್ಕೆ, ನೀರಾವರಿ, ಕೀಟ ನಿಯಂತ್ರಣ ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಧೋರಣೆಗಳ ಬಗ್ಗೆ ಸ್ಥಳೀಯ ಪರಿಸ್ಥಿತಿಗಳ ಆಧಾರದಲ್ಲಿ ವೈಯಕ್ತಿಕ ಸಲಹೆಗಳನ್ನು ನೀಡುತ್ತದೆ."
        features_section = "ವೈಶಿಷ್ಟ್ಯಗಳ ವಿಭಾಗ"
        key_features = "ಮುಖ್ಯ ವೈಶಿಷ್ಟ್ಯಗಳು"
        feature_1_title = "ಸ್ಮಾರ್ಟ್ ಹವಾಮಾನ ಒಳನೋಟಗಳು"
        feature_1_desc = "ನಿಖರ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ ಮತ್ತು ಎಚ್ಚರಿಕೆಗಳನ್ನು ಪಡೆದು ಕೃಷಿ ಕಾರ್ಯಗಳನ್ನು ಸಮರ್ಥವಾಗಿ ಯೋಜಿಸಿ."
        feature_2_title = "ಬೆಳೆ ಶಿಫಾರಸುಗಳು"
        feature_2_desc = "ಮಣ್ಣು, ಋತು ಮತ್ತು ಪ್ರದೇಶದ ಆಧಾರದ ಮೇಲೆ ಬೆಳೆಗಳಿಗೆ AI ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ."
        feature_3_title = "ರೋಗ ಪತ್ತೆ"
        feature_3_desc = "ಚಿತ್ರ ವಿಶ್ಲೇಷಣೆಯಿಂದ ಬೆಳೆ ರೋಗಗಳನ್ನು ಬೇಗ ಗುರುತಿಸಿ ಮತ್ತು ತಕ್ಷಣದ ಪರಿಹಾರ ಪಡೆಯಿರಿ."
        feature_4_title = "ಮಾರುಕಟ್ಟೆ ಬೆಲೆ ಮುನ್ನೋಟ"
        feature_4_desc = "ಮಂಡಿ ಬೆಲೆಗಳು ಮತ್ತು ಭವಿಷ್ಯದ ಧೋರಣೆಗಳನ್ನು ತಿಳಿದು ಸರಿಯಾದ ಸಮಯದಲ್ಲಿ ಮಾರಾಟ ಮಾಡಿ."
        feature_5_title = "ಬಹುಭಾಷಾ ಬೆಂಬಲ"
        feature_5_desc = "ಉತ್ತಮ ಅರಿವಿಗಾಗಿ ನಿಮ್ಮ ಇಷ್ಟದ ಸ್ಥಳೀಯ ಭಾಷೆಯಲ್ಲಿ ಎಲ್ಲಾ ವೈಶಿಷ್ಟ್ಯಗಳನ್ನು ಬಳಸಿ."
        how_it_works = "ಇದು ಹೇಗೆ ಕೆಲಸ ಮಾಡುತ್ತದೆ"
        how_it_works_title = "ನಮ್ಮ ವ್ಯವಸ್ಥೆ ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ"
        step_1 = "ನಿಮ್ಮ ಸ್ಥಳ ಮತ್ತು ಬೆಳೆ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ"
        step_2 = "AI ಹವಾಮಾನ, ಮಣ್ಣು ಮತ್ತು ಮಾರುಕಟ್ಟೆ ಡೇಟಾವನ್ನು ವಿಶ್ಲೇಷಿಸುತ್ತದೆ"
        step_3 = "ವೈಯಕ್ತಿಕ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಿರಿ"
        step_4 = "ಉತ್ಪಾದನೆ ಹೆಚ್ಚಿಸಿ ಮತ್ತು ಲಾಭ ಗರಿಷ್ಠಗೊಳಿಸಿ"
        why_choose = "ನಮ್ಮನ್ನು ಏಕೆ ಆಯ್ಕೆ ಮಾಡಬೇಕು"
        why_trust = "ರೈತರು ನಮ್ಮ ಮೇಲೆ ಏಕೆ ನಂಬಿಕೆ ಇಡುತ್ತಾರೆ"
        why_1 = "ಡೇಟಾ ಆಧಾರಿತ ಒಳನೋಟಗಳು"
        why_2 = "ಬಳಕೆ ಮಾಡಲು ಸುಲಭವಾದ ಇಂಟರ್ಫೇಸ್"
        why_3 = "ಮೊಬೈಲ್ ಮತ್ತು ಡೆಸ್ಕ್‌ಟಾಪ್‌ನಲ್ಲಿ ಕಾರ್ಯನಿರ್ವಹಿಸುತ್ತದೆ"
        why_4 = "ಗ್ರಾಮೀಣ ಸಂಪರ್ಕಕ್ಕೂ ಬೆಂಬಲ"
        why_5 = "ವಿಶ್ವಾಸಾರ್ಹ ಕೃಷಿ ಡೇಟಾ ಮೂಲಗಳು"
        footer_1 = "ಚತುರವಾಗಿ ಬೆಳೆಸಿ, ಉತ್ತಮವಾಗಿ ಕೊಯ್ಯಿರಿ"
        footer_2 = "ಮಣ್ಣಿನಿಂದ ಯಶಸ್ಸಿನವರೆಗೆ — ಡೇಟಾದ ಶಕ್ತಿ"
        footer_3 = "ನಿಮ್ಮ ಡಿಜಿಟಲ್ ಕೃಷಿ ಸಂಗಾತಿ"
        footer_4 = "ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಇಲ್ಲಿ ಆರಂಭವಾಗುತ್ತದೆ"
        footer_5 = "ರೈತರ ಸಬಲೀಕರಣ, ಸಮೃದ್ಧ ಭವಿಷ್ಯ"
        footer_6 = "खेती का स्मार्ट साथी (ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಂಗಾತಿ)"
        footer_7 = "ಉತ್ತಮ ನಾಳೆಗಾಗಿ ಡೇಟಾ ಆಧಾರಿತ ಕೃಷಿ"
    else:
        home_topbar_icon = "Home"
        home_topbar_title = "Farmer Advisory Home"
        online_status = "Online"
        home_badge = "Smart Agriculture Platform"
        home_hero_title = "Empowering Farmers with Smart Decisions"
        home_subtitle = "Get real-time crop advice, weather updates, disease alerts, and market prices — all in one place, in your local language."
        signal_title = "Live Agri Signals"
        signal_desc = "Weather alerts, crop support, and mandi trends in one colorful dashboard."
        local_lang_title = "Local Language Ready"
        local_lang_desc = "Easy guidance that farmers can understand and act on quickly."
        stat_1 = "AI Advisory"
        stat_2 = "Weather Signals"
        stat_3 = "Market Timing"
        register_label = "Register Farmer"
        registered_farmer_label = "Registered Farmer"
        name_label = "Name"
        place_label = "Place"
        mobile_label = "Mobile"
        language_label = "Language"
        registered_note = "You can now use the Farmer Query, Weather Prediction, and Price Prediction options."
        about_section = "About Section"
        about_title = "What is Farmer Advisory System?"
        about_desc = "Our Farmer Advisory System is an AI-powered platform designed to support farmers in making informed decisions. It provides personalized recommendations on crop selection, irrigation, pest control, and market trends based on real-time data and local conditions."
        features_section = "Features Section"
        key_features = "Key Features"
        feature_1_title = "Smart Weather Insights"
        feature_1_desc = "Get accurate weather forecasts and alerts to plan your farming activities efficiently."
        feature_2_title = "Crop Recommendations"
        feature_2_desc = "Receive AI-based suggestions on the best crops to grow based on soil, season, and region."
        feature_3_title = "Disease Detection"
        feature_3_desc = "Identify crop diseases early using image analysis and get instant treatment solutions."
        feature_4_title = "Market Price Forecasting"
        feature_4_desc = "Stay updated with mandi prices and future trends to sell your produce at the right time."
        feature_5_title = "Multilingual Support"
        feature_5_desc = "Access all features in your preferred local language for better understanding."
        how_it_works = "How It Works"
        how_it_works_title = "How Our System Helps You"
        step_1 = "Enter your location and crop details"
        step_2 = "AI analyzes weather, soil, and market data"
        step_3 = "Get personalized recommendations"
        step_4 = "Improve yield and maximize profit"
        why_choose = "Why Choose Us"
        why_trust = "Why Farmers Trust Us"
        why_1 = "Data-driven insights"
        why_2 = "Easy-to-use interface"
        why_3 = "Works on mobile & desktop"
        why_4 = "Supports rural connectivity"
        why_5 = "Trusted agricultural data sources"
        footer_1 = "Growing Smarter, Harvesting Better"
        footer_2 = "From Soil to Success — Powered by Data"
        footer_3 = "Your Digital Farming Companion"
        footer_4 = "Smart Farming Starts Here"
        footer_5 = "Empowering Farmers, Enriching Futures"
        footer_6 = "खेती का स्मार्ट साथी (Smart Farming Partner)"
        footer_7 = "Data-Driven Farming for a Better Tomorrow"

    st.markdown(
        f"""
        <div class="chat-topbar">
            <span class="topbar-icon">{home_topbar_icon}</span>
            <div class="topbar-text">
                {f'<img src="{home_topbar_logo}" alt="Farmer Advisory System" class="topbar-logo" />' if home_topbar_logo else f'<h2>{home_topbar_title}</h2>'}
            </div>
            <span class="topbar-status">{online_status}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="home-shell">
            <section class="home-hero">
                <div class="hero-grid">
                    <div>
                        <span class="home-badge">{home_badge}</span>
                        <h1>{home_hero_title}</h1>
                        <p class="home-sub">{home_subtitle}</p>
                    </div>
                    <div class="hero-right">
                        <div class="hero-mini-card">
                            <strong>{signal_title}</strong>
                            <span>{signal_desc}</span>
                        </div>
                        <div class="hero-mini-card">
                            <strong>{local_lang_title}</strong>
                            <span>{local_lang_desc}</span>
                        </div>
                        <div class="home-stat-row">
                            <div class="home-stat"><strong>24/7</strong><small>{stat_1}</small></div>
                            <div class="home-stat"><strong>Real-Time</strong><small>{stat_2}</small></div>
                            <div class="home-stat"><strong>Smart</strong><small>{stat_3}</small></div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.farmer_registered:
        with st.form("farmer_registration_form"):
            reg_name = st.text_input(_get_translation("farmer_name", st.session_state.response_language), placeholder=_get_translation("farmer_name_placeholder", st.session_state.response_language))
            reg_place = st.text_input(_get_translation("place_village", st.session_state.response_language), placeholder=_get_translation("place_placeholder", st.session_state.response_language))
            reg_mobile = st.text_input(_get_translation("mobile_no", st.session_state.response_language), placeholder=_get_translation("mobile_placeholder", st.session_state.response_language))
            reg_language = st.radio(
                _get_translation("language_label", st.session_state.response_language),
                ["English", "Kannada"],
                horizontal=True,
                index=0,
            )
            reg_submit = st.form_submit_button(register_label, use_container_width=True)

        if reg_submit:
            if not reg_name.strip() or not reg_place.strip() or not reg_mobile.strip():
                st.error(_get_translation("registration_incomplete", st.session_state.response_language))
            elif not re.fullmatch(r"\d{10}", reg_mobile.strip()):
                st.error(_get_translation("invalid_mobile", st.session_state.response_language))
            else:
                st.session_state.farmer_registered = True
                st.session_state.farmer_profile = {
                    "name": reg_name.strip(),
                    "place": reg_place.strip(),
                    "mobile": reg_mobile.strip(),
                    "language": reg_language,
                }
                st.session_state.response_language = reg_language
                st.success(_get_translation("registration_success", st.session_state.response_language).format(name=reg_name.strip()))
                st.rerun()
    else:
        profile = st.session_state.farmer_profile
        st.markdown(
            f"""
            <section class="home-section">
                <h3>{registered_farmer_label}</h3>
                <p><strong>{name_label}:</strong> {profile['name']}</p>
                <p><strong>{place_label}:</strong> {profile['place']}</p>
                <p><strong>{mobile_label}:</strong> {profile['mobile']}</p>
                <p><strong>{language_label}:</strong> {profile['language']}</p>
                <p>{registered_note}</p>
            </section>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <section class="home-section">
            <h3>{about_section}</h3>
            <p><strong>{about_title}</strong></p>
            <p>{about_desc}</p>
        </section>

        <section class="home-section">
            <h3>{features_section}</h3>
            <p><strong>{key_features}</strong></p>
            <div class="feature-grid">
                <div class="feature-card"><strong>{feature_1_title}</strong><span>{feature_1_desc}</span></div>
                <div class="feature-card"><strong>{feature_2_title}</strong><span>{feature_2_desc}</span></div>
                <div class="feature-card"><strong>{feature_3_title}</strong><span>{feature_3_desc}</span></div>
                <div class="feature-card"><strong>{feature_4_title}</strong><span>{feature_4_desc}</span></div>
                <div class="feature-card"><strong>{feature_5_title}</strong><span>{feature_5_desc}</span></div>
            </div>
        </section>

        <section class="home-section">
            <h3>{how_it_works}</h3>
            <p><strong>{how_it_works_title}</strong></p>
            <ul>
                <li>{step_1}</li>
                <li>{step_2}</li>
                <li>{step_3}</li>
                <li>{step_4}</li>
            </ul>
        </section>

        <section class="home-section">
            <h3>{why_choose}</h3>
            <p><strong>{why_trust}</strong></p>
            <ul>
                <li>{why_1}</li>
                <li>{why_2}</li>
                <li>{why_3}</li>
                <li>{why_4}</li>
                <li>{why_5}</li>
            </ul>
        </section>

        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="footer-rotator">
            <div class="footer-track">
                <span class="footer-pill">“{footer_1}”</span>
                <span class="footer-pill">“{footer_2}”</span>
                <span class="footer-pill">“{footer_3}”</span>
                <span class="footer-pill">“{footer_4}”</span>
                <span class="footer-pill">“{footer_5}”</span>
                <span class="footer-pill">“{footer_6}”</span>
                <span class="footer-pill">“{footer_7}”</span>

                <span class="footer-pill">“{footer_1}”</span>
                <span class="footer-pill">“{footer_2}”</span>
                <span class="footer-pill">“{footer_3}”</span>
                <span class="footer-pill">“{footer_4}”</span>
                <span class="footer-pill">“{footer_5}”</span>
                <span class="footer-pill">“{footer_6}”</span>
                <span class="footer-pill">“{footer_7}”</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif selected_page == "Farmer Query":
    if not st.session_state.farmer_registered:
        st.warning(_get_translation("not_registered_chat", st.session_state.response_language))
        st.stop()

    # Keep initial welcome message aligned with currently selected language.
    if st.session_state.messages:
        first = st.session_state.messages[0]
        if first.get("role") == "assistant" and first.get("content") in {
            _TRANSLATIONS["English"]["welcome"],
            _TRANSLATIONS["Kannada"]["welcome"],
        }:
            first["content"] = _get_translation("welcome", st.session_state.response_language)

    st.markdown(
        f"""
        <div class="chat-topbar">
            <span class="topbar-icon">🌾</span>
            <div class="topbar-text">
                <h2>{_get_translation("chat_topbar_title", st.session_state.response_language)}</h2>
                <p>{_get_translation("chat_topbar_subtitle", st.session_state.response_language)}</p>
            </div>
            <span class="topbar-status">{_get_translation("online_status", st.session_state.response_language)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_container = st.container(border=False)
    with chat_container:
        st.markdown("<div class='chat-scroll'>", unsafe_allow_html=True)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("audio"):
                    st.audio(message["audio"], format="audio/mp3")
        st.markdown("</div>", unsafe_allow_html=True)

    if mic_audio is not None:
        audio_bytes = mic_audio.getvalue()
        audio_hash = hashlib.md5(audio_bytes).hexdigest()
        if st.session_state.last_mic_hash != audio_hash:
            with st.spinner("Transcribing..."):
                try:
                    stt_language = "kn" if st.session_state.response_language == "Kannada" else "en"
                    transcribed = transcribe(mic_audio, language=stt_language)
                    st.session_state.last_mic_hash = audio_hash
                    if transcribed:
                        st.session_state.pending_prompt = transcribed
                        st.rerun()
                    else:
                        st.warning(_get_translation("speech_detection_failed", st.session_state.response_language))
                except Exception as exc:
                    st.session_state.last_mic_hash = audio_hash
                    st.error(_get_translation("speech_error", st.session_state.response_language).format(error=exc))

    # st.chat_input is Enter-to-send by default and includes a built-in send button.
    typed_prompt = st.chat_input(_get_translation("ask_question", st.session_state.response_language))

    prompt = typed_prompt or st.session_state.pending_prompt
    if prompt:
        display_text = st.session_state.pending_display_text or prompt
        st.session_state.pending_prompt = None
        st.session_state.pending_display_text = None

        st.session_state.messages.append({"role": "user", "content": display_text, "audio": None})
        with st.chat_message("user"):
            st.markdown(display_text)

        with st.chat_message("assistant"):
            valid, reason = _is_valid_query(prompt)
            if not valid:
                answer = _get_invalid_response(reason, st.session_state.response_language)
                final_answer = answer
            else:
                with st.spinner(_get_translation("preparing_advisory", st.session_state.response_language)):
                    answer = ask(prompt, response_language=st.session_state.response_language)
                expanded_question = _expand_farmer_question(prompt, st.session_state.response_language)
                final_answer = f"{expanded_question}\n\n{answer}"

            st.markdown(final_answer)

            answer_audio = None
            if st.session_state.enable_tts and final_answer:
                try:
                    answer_audio = speak_to_bytes(final_answer, lang="en")
                    st.audio(answer_audio, format="audio/mp3")
                except Exception as exc:
                    st.warning(_get_translation("voice_output_failed", st.session_state.response_language).format(error=exc))

        st.session_state.messages.append(
            {"role": "assistant", "content": final_answer, "audio": answer_audio}
        )
        st.rerun()

elif selected_page == "Weather Prediction":
    if not st.session_state.farmer_registered:
        st.warning(_get_translation("not_registered_weather", st.session_state.response_language))
        st.stop()

    st.markdown(
        """
        <div class="chat-topbar">
            <span class="topbar-icon">⛅</span>
            <div class="topbar-text">
                <h2>Weather Prediction</h2>
                <p>Enter your location to open weather insights for farming decisions</p>
            </div>
            <span class="topbar-status">Online</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(_get_translation("weather_title", st.session_state.response_language))
    with st.form("weather_form"):
        location = st.text_input(_get_translation("village_city", st.session_state.response_language), placeholder=_get_translation("village_placeholder", st.session_state.response_language))
        days = st.selectbox(_get_translation("forecast_window", st.session_state.response_language), [_get_translation("home", st.session_state.response_language).replace('ನೆಲೆ', '3 Days').replace('Home', "3 Days"), "7 Days", "10 Days"], index=1)
        weather_submit = st.form_submit_button("Open Weather Prediction", use_container_width=True)

    if weather_submit:
        if not location.strip():
            st.warning(_get_translation("location_required", st.session_state.response_language))
        else:
            day_count = int(days.split()[0])
            with st.spinner("Fetching live weather forecast..."):
                try:
                    weather = get_weather_forecast(location, day_count)
                    forecast = weather["forecast"]

                    st.success(_get_translation("forecast_loaded", st.session_state.response_language).format(location=weather['location_name'], days=day_count))

                    if forecast:
                        first = forecast[0]
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Today Max Temp", f"{first['temp_max']:.1f}°C")
                        c2.metric("Today Rain Chance", f"{first['rain_prob']:.0f}%")
                        c3.metric("Today Rainfall", f"{first['rain_mm']:.1f} mm")

                        st.markdown(_get_translation("daily_forecast", st.session_state.response_language))
                        st.dataframe(
                            [
                                {
                                    "Date": day["date"],
                                    "Temp Min (°C)": round(day["temp_min"], 1),
                                    "Temp Max (°C)": round(day["temp_max"], 1),
                                    "Rainfall (mm)": round(day["rain_mm"], 1),
                                    "Rain Probability (%)": round(day["rain_prob"], 0),
                                    "Wind (km/h)": round(day["wind_kmh"], 1),
                                }
                                for day in forecast
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown(_get_translation("farming_advisory", st.session_state.response_language))
                        st.info(_weather_advice(forecast))
                except Exception as exc:
                    st.error(_get_translation("weather_error", st.session_state.response_language).format(error=exc))

elif selected_page == "Price Prediction":
    if not st.session_state.farmer_registered:
        st.warning(_get_translation("not_registered_price", st.session_state.response_language))
        st.stop()

    st.markdown(
        """
        <div class="chat-topbar">
            <span class="topbar-icon">📈</span>
            <div class="topbar-text">
                <h2>Price Prediction</h2>
                <p>Select crop and market to open predicted mandi prices</p>
            </div>
            <span class="topbar-status">Online</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(_get_translation("price_title", st.session_state.response_language))
    with st.form("price_form"):
        crop = st.selectbox("Crop", ["Wheat", "Rice", "Maize", "Cotton", "Soybean", "Tomato", "Onion", "Potato"])
        mandi = st.selectbox("Market (Mandi)", ["Delhi", "Jaipur", "Lucknow", "Indore", "Nagpur"])
        location = st.text_input("Hyperlocal location (Village/City)", placeholder="e.g., Mysuru")
        current_price = st.number_input("Current market price (₹/qtl)", min_value=100.0, value=2200.0, step=50.0)
        quantity_qtl = st.number_input("Quantity to sell (qtl)", min_value=1.0, value=20.0, step=1.0)
        storage_cost_per_day = st.number_input("Storage cost (₹/qtl/day)", min_value=0.0, value=3.0, step=0.5)
        transport_cost = st.number_input("Transport cost total (₹)", min_value=0.0, value=1200.0, step=100.0)
        horizon = st.selectbox("Prediction horizon", ["7 Days", "14 Days", "30 Days"], index=0)
        price_submit = st.form_submit_button("Open Price Prediction", use_container_width=True)

    if price_submit:
        if not location.strip():
            st.warning("Please enter a location for weather-integrated advice.")
        else:
            owm_key = _get_secret_or_env("OPENWEATHER_API_KEY", "")

            if not owm_key:
                st.error("Missing OPENWEATHER_API_KEY. Add it in your .env and restart the app.")
            else:
                with st.spinner("Fusing price forecast with OpenWeatherMap + NASA POWER weather signals..."):
                    try:
                        price_engine = None
                        model_forecast = None
                        model_lower = None
                        model_upper = None
                        model_weights = None

                        try:
                            price_engine = get_engine_price_forecast(crop, mandi)
                            model_forecast = price_engine["forecast"]
                            model_lower = price_engine["lower_ci"]
                            model_upper = price_engine["upper_ci"]
                            model_weights = price_engine["weights"]
                        except Exception as model_exc:
                            st.warning(
                                "Advanced forecasting engine unavailable for this crop/mandi yet. "
                                "Using trend-based fallback for optimization. "
                                f"(Details: {model_exc})"
                            )

                        weather = _get_hyperlocal_weather(location, owm_key)
                        nasa_baseline = get_nasa_power_baseline(weather["lat"], weather["lon"])
                        risk = _fuse_weather_risk(crop, weather["forecast"], nasa_baseline)
                        result = optimize_sell_windows(
                            crop=crop,
                            current_price=float(current_price),
                            quantity_qtl=float(quantity_qtl),
                            storage_cost_per_day=float(storage_cost_per_day),
                            transport_cost=float(transport_cost),
                            weather_risk=risk,
                            model_forecast=model_forecast,
                        )

                        provider_label = "OpenWeatherMap" if weather.get("provider") == "openweathermap" else "Open-Meteo"
                        st.success(
                            f"Weather-integrated sell advisory for {crop} | {weather['location_name']} ({mandi})"
                        )
                        st.caption(f"Weather source: {provider_label} + NASA POWER baseline")

                        if model_forecast and model_lower and model_upper and model_weights:
                            st.markdown(_get_translation("crop_price_forecasting", st.session_state.response_language))
                            f7, f14, f30 = model_forecast[7], model_forecast[14], model_forecast[30]
                            c1p, c2p, c3p = st.columns(3)
                            c1p.metric("7-day price", f"₹{f7:,.0f}", help=f"95% CI: ₹{model_lower[7]:,.0f} - ₹{model_upper[7]:,.0f}")
                            c2p.metric("14-day price", f"₹{f14:,.0f}", help=f"95% CI: ₹{model_lower[14]:,.0f} - ₹{model_upper[14]:,.0f}")
                            c3p.metric("30-day price", f"₹{f30:,.0f}", help=f"95% CI: ₹{model_lower[30]:,.0f} - ₹{model_upper[30]:,.0f}")
                            st.caption(
                                "Model weights (optimized via CV): "
                                f"Prophet {model_weights['prophet']:.2f}, "
                                f"LSTM {model_weights['lstm']:.2f}, "
                                f"XGBoost {model_weights['xgboost']:.2f}"
                            )
                        st.markdown(_get_translation("alert_engine", st.session_state.response_language))
                        for alert in risk["alerts"]:
                            st.write(alert)

                        c1, c2, c3 = st.columns(3)
                        c1.metric("Weather Risk Score", f"{risk['score']:.0f}/100")
                        c2.metric("7-day Rain", f"{risk['total_rain_mm_7d']:.1f} mm")
                        c3.metric("Rain vs Baseline", f"{risk['rain_anomaly_mm']:+.1f} mm")

                        st.markdown(_get_translation("profit_optimizer", st.session_state.response_language))
                        st.dataframe(
                            [
                                {
                                    "Sell After (Days)": row["window_days"],
                                    "Expected Price (₹/qtl)": round(row["expected_price"], 2),
                                    "Storage Cost (₹)": round(row["storage_cost"], 2),
                                    "Transport Cost (₹)": round(row["transport_cost"], 2),
                                    "Weather Penalty (₹)": round(row["weather_penalty"], 2),
                                    "Net Income (₹)": round(row["net_income"], 2),
                                }
                                for row in result["rows"]
                            ],
                            hide_index=True,
                            use_container_width=True,
                        )

                        st.markdown(_get_translation("recommendation", st.session_state.response_language))
                        st.info(
                            (
                                f"**{result['recommendation']}**\n\n"
                                f"{result['rationale']}\n\n"
                                f"Expected income impact vs selling now: **₹{result['income_impact']:+,.0f}**\n\n"
                                f"Confidence score: **{(0.6 * result['confidence'] + 0.4 * (price_engine['confidence'] if price_engine else 70.0)):.0f}/100**"
                            )
                        )
                    except Exception as exc:
                        st.error(
                            "Sell advisor failed. Ensure price dataset exists at "
                            f"`{_PRICE_DATA_CSV}` with columns date,crop,mandi,price (optional: rainfall_index,msp_floor). "
                            f"Details: {exc}"
                        )
