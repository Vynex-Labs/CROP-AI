"""Presentation labels only. Do not rewrite underlying recommendations."""

from __future__ import annotations

LABELS = {
    "en": {
        "disease": "Disease",
        "risk": "Risk",
        "action": "Action",
        "referral": "Referral",
        "monitoring": "Monitoring",
        "follow_up": "Follow-up",
        "expert": "Extension / expert validation recommended",
        "laboratory": "Laboratory referral recommended",
        "chemical_blocked": "Chemical control not authorised by this system",
    },
    "hi": {
        "disease": "रोग",
        "risk": "जोखिम",
        "action": "कार्रवाई",
        "referral": "रेफरल",
        "monitoring": "निगरानी",
        "follow_up": "फॉलो-अप",
        "expert": "कृषि विस्तार / विशेषज्ञ सत्यापन अनुशंसित",
        "laboratory": "प्रयोगशाला रेफरल अनुशंसित",
        "chemical_blocked": "इस प्रणाली द्वारा रासायनिक नियंत्रण अधिकृत नहीं",
    },
    "mr": {
        "disease": "रोग",
        "risk": "धोका",
        "action": "कृती",
        "referral": "रेफरल",
        "monitoring": "निरीक्षण",
        "follow_up": "फॉलो-अप",
        "expert": "कृषी विस्तार / तज्ञ पडताळणी शिफारस",
        "laboratory": "प्रयोगशाळा रेफरल शिफारस",
        "chemical_blocked": "या प्रणालीने रासायनिक नियंत्रण अधिकृत नाही",
    },
}


def labels_for(language: str) -> dict[str, str]:
    lang = (language or "en").split("-")[0].lower()
    if lang not in LABELS:
        lang = "en"
    return dict(LABELS[lang])
