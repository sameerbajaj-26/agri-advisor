"""Language detection and fixed, human-written strings (alert text, disclaimer) in en/hi/kn.
Fixed strings keep safety messages identical whichever LLM answers. REVIEW hi/kn wording with a
native speaker / agri extension worker before launch."""

from typing import Literal

from app.agent.guards import Alert

Lang = Literal["en", "hi", "kn"]
LANG_NAMES: dict[Lang, str] = {"en": "English", "hi": "Hindi", "kn": "Kannada"}


def detect_language(text: str) -> Lang:
    """Majority Unicode script: Kannada U+0C80-0CFF, Devanagari U+0900-097F, else English."""
    kn = sum(1 for c in text if "\u0c80" <= c <= "\u0cff")
    hi = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    if kn == 0 and hi == 0:
        return "en"
    return "kn" if kn > hi else "hi"


DISCLAIMER: dict[Lang, str] = {
    "en": (
        "This is general information based on the sources shown and may not fit your exact field. "
        "Before spraying, harvesting or selling, confirm with your local Krishi Vigyan Kendra (KVK) "
        "or agriculture officer, and follow the instructions on the pesticide label."
    ),
    "hi": (
        "यह जानकारी दिखाए गए स्रोतों पर आधारित सामान्य जानकारी है और आपके खेत की स्थिति के अनुसार "
        "पूरी तरह सही नहीं भी हो सकती है। छिड़काव, कटाई या बिक्री से पहले अपने स्थानीय कृषि विज्ञान "
        "केंद्र (KVK) या कृषि अधिकारी से पुष्टि करें और कीटनाशक के लेबल पर दिए निर्देशों का पालन करें।"
    ),
    "kn": (
        "ಇದು ತೋರಿಸಿರುವ ಮೂಲಗಳ ಆಧಾರದ ಸಾಮಾನ್ಯ ಮಾಹಿತಿ ಆಗಿದ್ದು, ನಿಮ್ಮ ಹೊಲದ ನಿರ್ದಿಷ್ಟ ಪರಿಸ್ಥಿತಿಗೆ "
        "ಸಂಪೂರ್ಣವಾಗಿ ಹೊಂದಿಕೆಯಾಗದಿರಬಹುದು. ಸಿಂಪಡಣೆ, ಕೊಯ್ಲು ಅಥವಾ ಮಾರಾಟಕ್ಕೂ ಮೊದಲು ನಿಮ್ಮ ಸ್ಥಳೀಯ "
        "ಕೃಷಿ ವಿಜ್ಞಾನ ಕೇಂದ್ರ (KVK) ಅಥವಾ ಕೃಷಿ ಅಧಿಕಾರಿಯೊಂದಿಗೆ ಖಚಿತಪಡಿಸಿಕೊಳ್ಳಿ ಮತ್ತು ಕೀಟನಾಶಕದ "
        "ಲೇಬಲ್‌ನಲ್ಲಿರುವ ಸೂಚನೆಗಳನ್ನು ಪಾಲಿಸಿ."
    ),
}

DEGRADED: dict[Lang, str] = {
    "en": "I could not prepare a verified answer. Please read the safety alerts and sources below.",
    "hi": "मैं सत्यापित उत्तर तैयार नहीं कर सका। कृपया नीचे दी गई सुरक्षा चेतावनियाँ और स्रोत देखें।",
    "kn": "ನಾನು ಪರಿಶೀಲಿತ ಉತ್ತರವನ್ನು ಸಿದ್ಧಪಡಿಸಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ. ದಯವಿಟ್ಟು ಕೆಳಗಿನ ಸುರಕ್ಷತಾ ಎಚ್ಚರಿಕೆಗಳು ಮತ್ತು ಮೂಲಗಳನ್ನು ನೋಡಿ.",
}

NO_PRICE_DATA: dict[Lang, str] = {
    "en": "No mandi price data was retrieved, so there is no sell-or-hold recommendation.",
    "hi": "मंडी भाव का डेटा नहीं मिला, इसलिए बेचने या रोकने की कोई सलाह नहीं दी जा सकती।",
    "kn": "ಮಂಡಿ ಬೆಲೆ ಮಾಹಿತಿ ಸಿಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ಮಾರಾಟ ಅಥವಾ ಹಿಡಿದಿಡುವ ಬಗ್ಗೆ ಸಲಹೆ ನೀಡಲು ಸಾಧ್ಯವಿಲ್ಲ.",
}

ALERT_TEXT: dict[str, dict[Lang, str]] = {
    "rain_48h": {
        "en": "Rain expected in the next {hours} hours: about {total_mm} mm in total (highest hourly "
        "chance {max_prob}%). Do not spray or harvest until the rain has passed.",
        "hi": "अगले {hours} घंटों में बारिश की संभावना है: कुल लगभग {total_mm} मिमी (किसी एक घंटे में "
        "अधिकतम संभावना {max_prob}%)। बारिश गुज़रने तक छिड़काव और कटाई न करें।",
        "kn": "ಮುಂದಿನ {hours} ಗಂಟೆಗಳಲ್ಲಿ ಮಳೆಯ ಸಾಧ್ಯತೆ ಇದೆ: ಒಟ್ಟು ಸುಮಾರು {total_mm} ಮಿ.ಮೀ (ಯಾವುದೇ ಒಂದು "
        "ಗಂಟೆಯಲ್ಲಿ ಗರಿಷ್ಠ ಸಾಧ್ಯತೆ {max_prob}%). ಮಳೆ ಮುಗಿಯುವವರೆಗೆ ಸಿಂಪಡಣೆ ಮತ್ತು ಕೊಯ್ಲು ಮಾಡಬೇಡಿ.",
    },
    "heat": {
        "en": "High temperature expected: up to {max_c} °C within the next {hours} hours "
        "(alert level {threshold_c} °C).",
        "hi": "अगले {hours} घंटों में अधिक गर्मी की संभावना: तापमान {max_c} °C तक (चेतावनी सीमा "
        "{threshold_c} °C)।",
        "kn": "ಮುಂದಿನ {hours} ಗಂಟೆಗಳಲ್ಲಿ ಹೆಚ್ಚಿನ ಉಷ್ಣಾಂಶದ ಸಾಧ್ಯತೆ: {max_c} °C ವರೆಗೆ (ಎಚ್ಚರಿಕೆ ಮಟ್ಟ "
        "{threshold_c} °C).",
    },
    "frost": {
        "en": "Low temperature expected: down to {min_c} °C within the next {hours} hours "
        "(alert level {threshold_c} °C); frost damage is possible.",
        "hi": "अगले {hours} घंटों में कम तापमान की संभावना: {min_c} °C तक (चेतावनी सीमा {threshold_c} °C); "
        "पाले से नुकसान हो सकता है।",
        "kn": "ಮುಂದಿನ {hours} ಗಂಟೆಗಳಲ್ಲಿ ಕಡಿಮೆ ಉಷ್ಣಾಂಶದ ಸಾಧ್ಯತೆ: {min_c} °C ವರೆಗೆ (ಎಚ್ಚರಿಕೆ ಮಟ್ಟ "
        "{threshold_c} °C); ಹಿಮದಿಂದ ಹಾನಿಯಾಗಬಹುದು.",
    },
    "forecast_incomplete": {
        "en": "The forecast covers only {covered} of the next {hours} hours, so weather safety checks "
        "may miss risks. Be careful even if no other alert is shown.",
        "hi": "पूर्वानुमान अगले {hours} में से केवल {covered} घंटों को कवर करता है, इसलिए मौसम सुरक्षा जाँच में "
        "कुछ जोखिम छूट सकते हैं। कोई और चेतावनी न दिखे तब भी सावधानी रखें।",
        "kn": "ಮುನ್ಸೂಚನೆ ಮುಂದಿನ {hours} ಗಂಟೆಗಳಲ್ಲಿ ಕೇವಲ {covered} ಗಂಟೆಗಳನ್ನು ಒಳಗೊಂಡಿದೆ, ಆದ್ದರಿಂದ ಹವಾಮಾನ "
        "ಸುರಕ್ಷತಾ ಪರಿಶೀಲನೆ ಕೆಲವು ಅಪಾಯಗಳನ್ನು ತಪ್ಪಿಸಬಹುದು. ಬೇರೆ ಎಚ್ಚರಿಕೆ ಕಾಣದಿದ್ದರೂ ಜಾಗರೂಕರಾಗಿರಿ.",
    },
    "forecast_stale": {
        "en": "The weather data is {age_hours} hours old (the live source was unavailable). Check "
        "current conditions before acting.",
        "hi": "मौसम का डेटा {age_hours} घंटे पुराना है (लाइव स्रोत उपलब्ध नहीं था)। कार्य करने से पहले "
        "वर्तमान स्थिति जाँच लें।",
        "kn": "ಹವಾಮಾನ ಮಾಹಿತಿ {age_hours} ಗಂಟೆ ಹಳೆಯದು (ನೇರ ಮೂಲ ಲಭ್ಯವಿರಲಿಲ್ಲ). ಕ್ರಮ ಕೈಗೊಳ್ಳುವ ಮೊದಲು ಪ್ರಸ್ತುತ "
        "ಪರಿಸ್ಥಿತಿ ಪರಿಶೀಲಿಸಿ.",
    },
    "no_forecast": {
        "en": "No weather forecast could be retrieved for this question, so weather safety checks were "
        "NOT run. Do not assume the weather is safe.",
        "hi": "इस प्रश्न के लिए मौसम का पूर्वानुमान प्राप्त नहीं हो सका, इसलिए मौसम सुरक्षा जाँच नहीं हुई। "
        "मौसम को सुरक्षित न मानें।",
        "kn": "ಈ ಪ್ರಶ್ನೆಗೆ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ ಪಡೆಯಲಾಗಲಿಲ್ಲ, ಆದ್ದರಿಂದ ಹವಾಮಾನ ಸುರಕ್ಷತಾ ಪರಿಶೀಲನೆ ನಡೆಯಲಿಲ್ಲ. "
        "ಹವಾಮಾನ ಸುರಕ್ಷಿತವೆಂದು ಭಾವಿಸಬೇಡಿ.",
    },
}


def localize_alerts(alerts: list[Alert], lang: Lang) -> list[Alert]:
    out: list[Alert] = []
    for a in alerts:
        template = ALERT_TEXT[a.code].get(lang) or ALERT_TEXT[a.code]["en"]
        out.append(a.model_copy(update={"message": template.format(**a.params)}))
    return out
