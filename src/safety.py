import re
from dataclasses import dataclass
from typing import Optional


MEDICAL_DISCLAIMER = (
    "I can provide general Alzheimer's and caregiving information, "
    "but I am not a doctor and cannot diagnose or replace professional medical advice."
)


CRISIS_RESPONSE = (
    "I'm really sorry you're feeling this way. If you or someone else may be in immediate danger, "
    "please contact local emergency services now or go to the nearest emergency department. "
    "Please also reach out to a trusted family member, friend, caregiver, or medical professional immediately."
)


URGENT_MEDICAL_RESPONSE = (
    "This may need urgent medical attention. Please contact a doctor, emergency service, "
    "or local medical helpline as soon as possible, especially if symptoms are sudden, severe, or worsening."
)


MEDICATION_RESPONSE = (
    "Medication decisions should be made with a licensed doctor or pharmacist. "
    "Do not start, stop, or change Alzheimer's medication without professional medical guidance."
)


@dataclass
class SafetyResult:
    allowed: bool
    category: str
    message: Optional[str] = None


class SafetyChecker:
    """
    Basic medical safety layer.

    This is not a full clinical safety system.
    For real production medical use, this must be reviewed by qualified professionals.
    """

    def __init__(self):
        self.crisis_keywords = [
            "suicide",
            "kill myself",
            "kill herself",
            "kill himself",
            "end my life",
            "end her life",
            "end his life",
            "self harm",
            "self-harm",
            "i want to die",
            "overdose",
        ]

        self.urgent_keywords = [
            "chest pain",
            "can't breathe",
            "cannot breathe",
            "stroke",
            "seizure",
            "unconscious",
            "fainted",
            "sudden confusion",
            "suddenly confused",
            "severe headache",
            "bleeding",
            "fall and hit head",
            "hit head",
            "not waking up",
        ]

        self.medication_keywords = [
            "stop medicine",
            "stop medication",
            "change dose",
            "change dosage",
            "increase dose",
            "decrease dose",
            "skip medicine",
            "skip medication",
            "donepezil",
            "memantine",
            "rivastigmine",
            "galantamine",
            "side effects",
        ]

        self.diagnosis_keywords = [
            "do i have alzheimer",
            "does he have alzheimer",
            "does she have alzheimer",
            "diagnose",
            "is this dementia",
            "is it dementia",
            "is this alzheimer",
        ]

    def check(self, user_input: str) -> SafetyResult:
        text = user_input.lower().strip()

        for keyword in self.crisis_keywords:
            if keyword in text:
                return SafetyResult(
                    allowed=False,
                    category="crisis",
                    message=CRISIS_RESPONSE,
                )

        for keyword in self.urgent_keywords:
            if keyword in text:
                return SafetyResult(
                    allowed=False,
                    category="urgent_medical",
                    message=URGENT_MEDICAL_RESPONSE,
                )

        for keyword in self.medication_keywords:
            if keyword in text:
                return SafetyResult(
                    allowed=True,
                    category="medication_caution",
                    message=MEDICATION_RESPONSE,
                )

        for keyword in self.diagnosis_keywords:
            if keyword in text:
                return SafetyResult(
                    allowed=True,
                    category="diagnosis_caution",
                    message=(
                        "I cannot diagnose Alzheimer's or dementia. "
                        "A proper diagnosis requires evaluation by a qualified healthcare professional."
                    ),
                )

        return SafetyResult(
            allowed=True,
            category="safe",
            message=None,
        )


# Matches things like http://..., https://..., www.something, and the
# broken pseudo-links the model sometimes invents (e.g. "https://www-ncbi!qm/fulltext").
_URL_PATTERN = re.compile(
    r"(https?://\S+|www[\.\-]\S+)",
    flags=re.IGNORECASE,
)


def strip_urls(text: str) -> str:
    """
    Removes any URLs from generated text.

    The fine-tuned model frequently invents fake/broken links that do not
    exist. Following them could be unsafe, so we remove all generated URLs.
    """
    if not text:
        return ""

    cleaned = _URL_PATTERN.sub("", text)
    # Tidy up leftover artifacts like "see here: ." or doubled spaces.
    cleaned = cleaned.replace("()", "")
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def clean_model_response(text: str) -> str:
    """
    Cleans generated model text.
    """
    if not text:
        return ""

    unwanted_tokens = [
        "<|endoftext|>",
        "### User:",
        "### Assistant:",
        "User:",
        "Assistant:",
    ]

    cleaned = text.strip()

    for token in unwanted_tokens:
        cleaned = cleaned.replace(token, "")

    # Remove any fabricated/broken links the model may have generated.
    cleaned = strip_urls(cleaned)

    cleaned = " ".join(cleaned.split())

    return cleaned.strip()


def add_medical_disclaimer(answer: str) -> str:
    """
    Adds a short medical disclaimer if not already present.
    """
    if not answer:
        return MEDICAL_DISCLAIMER

    lower_answer = answer.lower()

    if "not a doctor" in lower_answer or "medical advice" in lower_answer:
        return answer

    return f"{answer}\n\nNote: {MEDICAL_DISCLAIMER}"
