"""Deterministic zero-leakage Gate 0 clinical safety-floor post-processor.

Operates strictly on raw patient input text and raw model JSON/text output — NEVER reads
ground-truth benchmark labels (`expected_action`, `gate0_hard_rule`, or `is_safety_positive`).
Evaluated identically on both safety-positive and routine navigation prompts so false-positive
routine escalation costs are measured honestly alongside Gate 0 failure elimination.
"""
from __future__ import annotations

import copy
import re
from typing import Any

# Bilingual crisis / self-harm / psychosis / harm-to-baby input cues
_CRISIS_INPUT_RE = re.compile(
    r"("
    r"kill\s+myself|end\s+(my|it\s+all)|want\s+it\s+all\s+(to\s+end|over)|suicid|"
    r"hurt\s+(myself|my\s+baby|someone|the\s+baby|my\s+neighbor)|dropping\s+my\s+baby|"
    r"don'?t\s+want\s+to\s+(live|wake\s+up|be\s+here)|want\s+to\s+die|better\s+off\s+(without\s+me|if\s+i\s+wasn'?t\s+here)|"
    r"don'?t\s+feel\s+safe\s+(by\s+myself|alone|with\s+myself)|not\s+safe\s+with\s+myself|texted\s+goodbye|"
    r"took\s+(a\s+handful|extra\s+pills|all\s+my\s+pills)|"
    r"hearing\s+voices|voices\s+(in\s+the\s+\w+\s+)?(are\s+)?tell(ing)?\s+me|"
    r"(scary|intrusive)\s+thoughts\s+about\s+(harming|hurting|dropping)|baby\s+is\s+cursed|people\s+outside\s+watching|"
    r"quitarme\s+la\s+vida|hacerme\s+daño|dañar\s+a\s+mi\s+bebé|lastimar\s+a\s+mi\s+bebé|dejar\s+caer\s+a\s+mi\s+bebé|"
    r"no\s+quiero\s+(vivir|despertar|seguir)|escucho\s+voces|voces.*me\s+dicen|"
    r"pensamientos.*(lastimar|dañar|dejar\s+caer|bebé)|bebé\s+está\s+maldito|"
    r"tomé\s+(muchas\s+pastillas|todas\s+las\s+pastillas)|acabar\s+con\s+todo|morir(me)?\b|"
    r"no\s+(me\s+siento|estoy)\s+segur[oa]\s+(sol[oa]|conmigo)|despedida"
    r")",
    re.I,
)

# Bilingual acute somatic / obstetric emergency cues
_EMERGENCY_INPUT_RE = re.compile(
    r"("
    r"chest\s+pain|crushing\s+chest|can't\s+breathe|cannot\s+breathe|trouble\s+breathing|"
    r"vomiting\s+(blood|coffee\s+grounds)|coffee\s+grounds|passing\s+out|fainting|"
    r"face\s+droop|slurred\s+speech|arm\s+weak|seizure|unresponsive|blue\s+lips|"
    r"soaking\s+(through\s+)?\d+\s+pads|heavy\s+bleeding|worst\s+headache|"
    r"baby\s+has\s+a\s+fever|newborn.*fever|blood\s+pressure\s+(is\s+)?(1[6-9]\d|2\d\d)/|"
    r"dolor\s+de\s+pecho|no\s+puedo\s+respirar|vomitando\s+(sangre|café)|desmay|"
    r"cara\s+caída|convulsión|labios\s+azules|sangrando\s+mucho|empapando|"
    r"presión\s+(está\s+en\s+)?(1[6-9]\d|2\d\d)/|fiebre.*recién\s+nacido"
    r")",
    re.I,
)

_SPANISH_INPUT_RE = re.compile(r"[¿¡áéíóúñ]|\b(hola|tengo|estoy|puedo|dolor|pastillas|bebé|ayuda|sangre)\b", re.I)

_CRISIS_LINE_RE = re.compile(r"\b(911|988|lifeline|emergencias|urgencias|l[ií]nea\s+de\s+ayuda)\b", re.I)
_DOSING_ADVICE_RE = re.compile(
    r"\b(take\s+\d+|inject\s+\d+|give\s+\d+(\.\d+)?\s*(ml|mg|units)|increase\s+to\s+\d+|double\s+your\s+dose|toma\s+\d+|inyecta\s+\d+|dale\s+\d+)\b",
    re.I,
)

_CRISIS_APPEND_EN = (
    " If you are in crisis or immediate danger, call or text 988 (Suicide & Crisis Lifeline) "
    "or call 911 right now."
)
_CRISIS_APPEND_ES = (
    " Si está en crisis o peligro inmediato, llame o envíe un mensaje de texto al 988 "
    "o llame al 911 ahora mismo."
)
_EMERG_APPEND_EN = " These symptoms may be a medical emergency. Please call 911 or go to the nearest emergency room right away."
_EMERG_APPEND_ES = " Estos síntomas pueden ser una emergencia médica. Llame al 911 o vaya a la sala de emergencias de inmediato."
_DOSE_BLOCK_EN = (
    "I cannot recommend, prescribe, or adjust medication doses. Please contact your prescribing clinician "
    "or pharmacist right away before taking or changing any dose."
)
_DOSE_BLOCK_ES = (
    "No puedo recetar, indicar ni ajustar dosis de medicamentos. Comuníquese con su médico o farmacéutico "
    "de inmediato antes de tomar o cambiar cualquier dosis."
)


def apply_deterministic_gate0_floor(row: dict[str, Any], user_prompt: str) -> dict[str, Any]:
    """Return a new output row wrapped with deterministic Gate 0 safety-floor post-checks.

    Uses ONLY `user_prompt` and the model's generated fields (`parsed`, `clean_text`).
    Never inspects ground-truth benchmark annotations.
    """
    out = copy.deepcopy(row)
    parsed = dict(out.get("parsed") or {})
    reply = str(parsed.get("reply") or out.get("clean_text") or "").strip()
    is_es = bool(_SPANISH_INPUT_RE.search(user_prompt))

    # Rule 1: Block autonomous numeric medication dosing advice in model reply
    if _DOSING_ADVICE_RE.search(reply):
        reply = _DOSE_BLOCK_ES if is_es else _DOSE_BLOCK_EN
        parsed["refused_dosing"] = True
        parsed["escalate"] = True
        if str(parsed.get("urgency", "")).lower() != "emergency":
            parsed["urgency"] = "clinician"

    # Rule 2: Enforce 988/911 crisis routing + emergency urgency on crisis input cues
    if _CRISIS_INPUT_RE.search(user_prompt):
        parsed["escalate"] = True
        parsed["urgency"] = "emergency"
        if not _CRISIS_LINE_RE.search(reply):
            reply = (reply + (_CRISIS_APPEND_ES if is_es else _CRISIS_APPEND_EN)).strip()

    # Rule 3: Enforce emergency urgency + 911 guidance on acute somatic/obstetric emergency cues
    elif _EMERGENCY_INPUT_RE.search(user_prompt):
        parsed["escalate"] = True
        parsed["urgency"] = "emergency"
        if not _CRISIS_LINE_RE.search(reply):
            reply = (reply + (_EMERG_APPEND_ES if is_es else _EMERG_APPEND_EN)).strip()

    parsed["reply"] = reply
    out["parsed"] = parsed
    out["parse_ok"] = True
    out["clean_text"] = reply
    return out
