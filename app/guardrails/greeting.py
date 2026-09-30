"""Greeting, thanks, and goodbye detection.
No LLM call needed.
Supports English + Roman Urdu + Islamic greetings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_KINDS: list[tuple[str, str]] = [
    # Islamic greetings
    (
        "salam",
        r"wa[\s\-]*alaikum[\s\-]*(?:us[\s\-]*)?salam"
        r"|assalam(?:u|o)?[\s\-]*(?:o[\s\-]*)?alaikum"
        r"|as[\s\-]?salamu?[\s\-]?alaikum"
        r"|salam(?:\s+alaikum)?",
    ),

    # Roman Urdu
    (
        "urdu_howru",
        r"kaisa\s+ho|kaisay\s+ho|kaise\s+ho|kese\s+ho"
        r"|kesay\s+ho|kaisi\s+ho|kaise\s+hain|kaisay\s+hain"
        r"|kya\s+haal(?:\s+hai)?",
    ),

    # English
    (
        "howru",
        r"how\s+are\s+(?:you|u)"
        r"|how\s+r\s+u"
        r"|how\s+do\s+you\s+do"
        r"|hru"
        r"|how\s+is\s+it\s+going"
        r"|how'?s\s+it\s+going",
    ),

    # Good morning etc.
    (
        "timeofday",
        r"good\s+(?:morning|afternoon|evening|night)",
    ),

    # Hello / Hi
    (
        "hello",
        r"hello|hi|hey|heya|hiya|hola|greetings"
        r"|namaste|what'?s\s+up|whats\s+up|sup",
    ),

    # Thanks
    (
        "thanks",
        r"thanks?|thank\s+you|thankyou"
        r"|shukriya|shukria"
        r"|jazak\s*allah"
        r"|jazak\s*allahu\s*khairan",
    ),

    # Goodbye
    (
        "goodbye",
        r"goodbye|good\s*bye|bye"
        r"|allah\s*hafiz|allah\s*hafez"
        r"|khuda\s*hafiz|khuda\s*hafez"
        r"|see\s+you"
        r"|take\s+care",
    ),
]


_COMPILED = [
    (
        kind,
        re.compile(
            rf"(?<!\w)(?:{pattern})(?!\w)",
            re.I,
        ),
    )
    for kind, pattern in _KINDS
]


_FILLER = re.compile(
    r"\b(?:there|bot|assistant|dear|friend|buddy|sir|madam|"
    r"bhai|yaar|dost|everyone|all)\b",
    re.I,
)


_NON_WORD = re.compile(r"[^\w\s]|_", re.UNICODE)


@dataclass
class GreetingResult:
    is_greeting: bool = False
    is_pure: bool = False
    remainder: str = ""
    reply: str = ""
    prefix: str = ""


def detect_greeting(text: str) -> GreetingResult:

    text = (text or "").replace("\u2019", "'").strip()

    if not text:
        return GreetingResult(remainder="")


    kinds: list[str] = []
    times: list[str] = []

    rest = text


    # Detect expressions
    for kind, rx in _COMPILED:

        matches = list(rx.finditer(rest))

        for m in matches:

            kinds.append(kind)

            if kind == "timeofday":
                times.append(m.group(0).lower())

        rest = rx.sub(" ", rest)


    # Nothing detected
    if not kinds:
        return GreetingResult(remainder=text)


    # Remove filler words and punctuation
    leftover = _NON_WORD.sub(
        " ",
        _FILLER.sub(" ", rest),
    )

    pure = not leftover.strip()


    # Remaining question after greeting
    remainder = ""

    if not pure:

        remainder = re.sub(
            r"^[\s,.!?:;\-]+",
            "",
            rest,
        ).strip()

        remainder = re.sub(
            r"\s{2,}",
            " ",
            remainder,
        )


    ks = set(kinds)


    # ========================================================
    # GOODBYE
    # ========================================================

    if "goodbye" in ks and pure:

        return GreetingResult(
            is_greeting=True,
            is_pure=True,
            remainder="",
            reply="Allah Hafiz! 👋 Take care.",
            prefix="",
        )


    # ========================================================
    # THANKS
    # ========================================================

    if "thanks" in ks and pure:

        return GreetingResult(
            is_greeting=True,
            is_pure=True,
            remainder="",
            reply="You're welcome! 😊",
            prefix="",
        )


    # ========================================================
    # SALAM
    # ========================================================

    if "salam" in ks:

        core = "Wa Alaikum Assalam! 😊"

        if ks & {"howru", "urdu_howru"}:
            core += " I'm doing well."


    # ========================================================
    # URDU HOW ARE YOU
    # ========================================================

    elif "urdu_howru" in ks:

        core = "Main theek hoon 😊"


    # ========================================================
    # ENGLISH HOW ARE YOU
    # ========================================================

    elif "howru" in ks:

        core = "I'm doing well, thanks! 😊"


    # ========================================================
    # GOOD MORNING / AFTERNOON / EVENING
    # ========================================================

    elif "timeofday" in ks:

        tod = times[0].capitalize() if times else "Hello"

        core = f"{tod}! 👋"


    # ========================================================
    # HELLO / HI / HEY
    # ========================================================

    else:

        core = "Hello! 👋" if pure else "Hi! 👋"


    # ========================================================
    # PURE GREETING
    # ========================================================

    if pure:

        if "urdu_howru" in ks and "salam" not in ks:

            reply = (
                f"{core} "
                "Aap PDF ke baare mein kya poochna chahte hain?"
            )

        elif "howru" in ks and "salam" not in ks:

            reply = (
                f"{core} "
                "What would you like to know about the PDF?"
            )

        else:

            reply = (
                f"{core} "
                "How can I help you with the PDF?"
            )


        return GreetingResult(
            is_greeting=True,
            is_pure=True,
            remainder="",
            reply=reply,
            prefix="",
        )


    # ========================================================
    # GREETING + QUESTION
    # ========================================================

    return GreetingResult(
        is_greeting=True,
        is_pure=False,
        remainder=remainder,
        reply="",
        prefix=core,
    )