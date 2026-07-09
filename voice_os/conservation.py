"""Content-conservation checks for redraft work.

The QA gate scores voice alignment; nothing in it asks whether the
output *says what the input said*. The 2026-07-08 field report
(feedback/2026-07-08-storytellermitch-site-pass.md) documented the
cost: a wholesale re-composition that invented four opinions passed at
fidelity 0.718 with no warning. These checks close that gap.

Everything here is stdlib-only and deterministic (the project-wide
offline contract): a lexical claims diff, quote-span inviolability,
protected qualifiers around facts, format mirroring, and a
diction-escalation diff. The product graph runs them in qa_gate and
surfaces the results in an additive envelope field; quote violations
always block a pass, unsupported sentences block only in redraft mode.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- lexical core

# Function words plus verbs so common they carry no claim content. A word
# on this list never counts toward (or against) sentence support.
_STOPWORDS = frozenset(
    """
    a an the and or but nor so yet for of in on at by to from with without
    into onto over under between among through during before after above
    below up down out off again further then once here there when where why
    how all any both each few more most other some such no not only own same
    than too very s t can will just don should now i you he she it we they
    me him her us them my your his its our their mine yours hers ours theirs
    this that these those am is are was were be been being have has had
    having do does did doing would could ought im youre hes shes its were
    theyre ive youve weve theyve id youd hed shed wed theyd ill youll hell
    shell well theyll isnt arent wasnt werent hasnt havent hadnt doesnt dont
    didnt wont wouldnt shant shouldnt cant cannot couldnt mustnt lets thats
    whos whats heres theres whens wheres whys hows because as until while if
    then else about against what which who whom whose one two also get got
    make makes made making go goes went going come comes came coming thing
    things way ways lot bit really actually
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(word: str) -> str:
    """Inflection-lite stem so 'tools'/'tooling' support 'tool'.

    Deliberately crude: strip one common suffix with a length guard.
    Both sides of every comparison pass through the same stem, so the
    only requirement is consistency, not linguistic correctness.
    """
    for suffix in ("ing", "ed", "es", "s", "ly"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("’", "'"))


def content_words(text: str) -> list[str]:
    """Stemmed content words, in order, stopwords removed."""
    return [
        _stem(w.replace("'", ""))
        for w in _tokens(text)
        if w.replace("'", "") not in _STOPWORDS and len(w) >= 3
    ]


def _supported(word: str, vocab: set[str]) -> bool:
    """A word is supported when the input vocabulary contains it, or
    contains a form sharing a 6-character prefix (architectural ->
    architecture and similar derivations the stem misses)."""
    if word in vocab:
        return True
    if len(word) >= 6:
        prefix = word[:6]
        return any(v.startswith(prefix) for v in vocab)
    return False


def split_sentences(text: str) -> list[str]:
    """Sentence split on terminal punctuation and blank lines."""
    parts = re.split(r"(?<=[.!?])[\"'”’)]*\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------- claims diff

# A sentence needs this many content words before the claims diff will
# judge it (shorter fragments are cadence, not claims), and is flagged
# when less than this fraction of them trace back to the input.
_MIN_CONTENT_WORDS = 3
_SUPPORT_THRESHOLD = 0.5


def _replacement_vocab() -> set[str]:
    """Content words of the house banned-phrase substitutions.

    The offline persona rewrites "please don't hesitate to reach out"
    to "let me know" from qa.REPLACEMENTS; the pipeline's own curated
    substitutions are never invented content, so their vocabulary is
    always supported."""
    from .qa import REPLACEMENTS

    vocab: set[str] = set()
    for value in REPLACEMENTS.values():
        vocab.update(content_words(value))
        vocab.update(_stem(w) for w in _tokens(value))
    return vocab


def unsupported_sentences(input_text: str, output_text: str) -> list[dict]:
    """Output sentences whose content the input does not entail.

    Lexical entailment: a sentence is unsupported when fewer than half
    of its content words appear in the input (stem or 6-char-prefix
    match). Calibrated against the 2026-07-08 site pass: every invented
    opinion in the report's receipts scores below 0.45 support, while
    the 17 human-approved rewrites of the same session produce no
    false flags at this threshold.
    """
    vocab = set(content_words(input_text)) | _replacement_vocab()
    flagged = []
    for sentence in split_sentences(output_text):
        words = content_words(sentence)
        if len(words) < _MIN_CONTENT_WORDS:
            continue
        support = sum(1 for w in words if _supported(w, vocab)) / len(words)
        if support < _SUPPORT_THRESHOLD:
            flagged.append({"sentence": sentence, "support": round(support, 3)})
    return flagged


# ---------------------------------------------------------------- quote spans

_QUOTE_SPAN = re.compile(r"\"[^\"\n]+\"")


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("“", '"').replace("”", '"')
        .replace("‘", "'").replace("’", "'")
    )


def quoted_spans(text: str) -> list[str]:
    """Double-quoted spans (straight or curly), marks included."""
    return _QUOTE_SPAN.findall(_normalize_quotes(text))


def quote_violations(input_text: str, output_text: str) -> list[str]:
    """Input quote spans that do not survive verbatim (marks included).

    Text inside quotation marks is someone's words; the pipeline must
    never touch it. Comparison normalizes curly/straight glyphs and is
    otherwise exact.
    """
    output_normalized = _normalize_quotes(output_text)
    return [
        span for span in quoted_spans(input_text)
        if span not in output_normalized
    ]


# ---------------------------------------------------------------- modifiers

# Framing labels: words that tell the reader how to weigh the adjacent
# fact. Dropping one changes the claim's epistemic status.
_FRAMING_LABELS = (
    "on air",
    "on the air",
    "on the record",
    "off the record",
    "on camera",
    "design targets",
    "design target",
    "internal figure",
    "internal figures",
    "internal estimate",
    "self-reported",
    "unverified",
    "by my count",
)

_NUMBER_WORDS = frozenset(
    """
    zero one two three four five six seven eight nine ten eleven twelve
    thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty
    thirty forty fifty sixty seventy eighty ninety hundred thousand
    million billion dozen half quarter
    """.split()
)

# Hedges that adjudicate a number's precision. Position: immediately
# before the numeral ("roughly fifty" — the hedge WAS the fact).
_NUMERAL_HEDGES = frozenset(
    """
    roughly about around approximately nearly almost some over under
    estimated maybe perhaps
    """.split()
)

# Words too generic to be a load-bearing modifier of a numeral.
_WINDOW_SKIP = _STOPWORDS | {"than", "least", "per"}

_TOKEN_SPLIT = re.compile(r"[^\w'’-]+")


def _is_numeric_token(token: str) -> bool:
    core = token.lower().strip("'’")
    if any(ch.isdigit() for ch in core):
        return True
    return any(part in _NUMBER_WORDS for part in core.split("-") if part)


def dropped_modifiers(input_text: str, output_text: str) -> list[dict]:
    """Qualifiers adjacent to facts that did not survive the rewrite.

    Two detectors:
    - numeral windows: a precision hedge within two tokens before a
      numeral ("roughly fifty" — the hedge was the adjudicated fact) and
      the modifier position directly after it ("four-month electrical
      blackout") must survive the rewrite while the numeral does;
    - framing labels: fixed phrases that label how a fact was sourced or
      delivered ("on air", "design targets", "internal figure").

    The number-checker already owns the numerals themselves; this owns
    the words around them. Advisory: the caller (and the revision loop)
    gets a diff, the gate decision does not flip on it. Verbs before a
    numeral ("reached 50") are deliberately outside scope: rephrasing
    them is legitimate revision, measured against the 2026-07-08 site
    pass where they were the only false positives.
    """
    flagged: list[dict] = []
    seen: set[tuple[str, str]] = set()

    output_lower = output_text.lower()
    output_vocab = set(content_words(output_text))

    def _bare(token: str) -> str:
        word = token.lower().strip("'’-").replace("’", "'")
        return re.sub(r"[^a-z0-9-]", "", word)

    def _missing(bare: str) -> bool:
        if not bare or bare in _WINDOW_SKIP or _is_numeric_token(bare):
            return False
        if _supported(_stem(bare.replace("-", "")), output_vocab):
            return False
        return bare.replace("-", " ") not in output_lower

    def _flag(bare: str, anchor: str) -> None:
        key = (bare, anchor.lower())
        if key not in seen:
            seen.add(key)
            flagged.append({
                "modifier": bare,
                "anchor": anchor,
                "kind": "numeral-adjacent",
            })

    def _scan_sentence(input_tokens: list[str]) -> None:
        for index, token in enumerate(input_tokens):
            if not _is_numeric_token(token):
                continue
            # The numeral itself must survive for a modifier check to
            # make sense; a lost numeral is the number-checker's finding.
            if _stem(token.lower().strip("'’-")) not in output_vocab and \
                    token.lower() not in output_lower:
                continue
            for neighbor in input_tokens[max(0, index - 2): index]:
                bare = _bare(neighbor)
                # Hedges bypass _missing's stopword skip ("about" is both
                # a stopword and, before a numeral, a load-bearing hedge).
                if bare in _NUMERAL_HEDGES and \
                        not re.search(r"\b" + re.escape(bare) + r"\b",
                                      output_lower):
                    _flag(bare, token)
            after = input_tokens[index + 1: index + 2]
            if after:
                bare = _bare(after[0])
                if _missing(bare):
                    _flag(bare, token)

    # Windows never cross a sentence boundary: a word ending one
    # sentence is not a modifier of a numeral opening the next.
    for sentence in split_sentences(input_text):
        _scan_sentence([t for t in _TOKEN_SPLIT.split(sentence) if t])

    input_lower = input_text.lower()
    for label in _FRAMING_LABELS:
        if label in input_lower and label not in output_lower:
            key = (label, "framing")
            if key not in seen:
                seen.add(key)
                flagged.append({
                    "modifier": label,
                    "anchor": None,
                    "kind": "framing-label",
                })
    return flagged


# ---------------------------------------------------------------- format

_MARKDOWN_LINE = re.compile(r"^\s*(?:[-*•]\s+|\d+\.\s+|#{1,6}\s+)")


def _markdown_lines(text: str) -> int:
    return sum(1 for line in text.splitlines() if _MARKDOWN_LINE.match(line))


def format_flags(input_text: str, output_text: str) -> list[str]:
    """Output formatting the input did not have (markdown into prose)."""
    flags = []
    if _markdown_lines(output_text) > _markdown_lines(input_text):
        flags.append(
            "markdown list or heading lines introduced; the input is prose "
            "and the output format must mirror the input format"
        )
    return flags


# ---------------------------------------------------------------- diction

# Charged terms the reviser reaches for when punching up. Flagged only
# when the output introduces one the input did not contain: that diff is
# exactly the escalation the field report documented ("pursuing its
# critics with legal tools" -> "hunting its critics").
_CHARGED_TERMS = (
    "hunt", "hunting", "hunted",
    "war", "warfare", "weapon", "weaponize", "weaponized",
    "lethal", "deadly",
    "destroy", "destroyed", "destroying",
    "crush", "crushed", "crushing",
    "attack", "attacked", "attacking", "assault",
    "brutal", "savage", "vicious", "ruthless",
    "gossip",
    "slaughter", "annihilate",
)


def escalated_diction(input_text: str, output_text: str) -> list[str]:
    """Charged terms present in the output but absent from the input."""
    input_stems = {_stem(w) for w in _tokens(input_text)}
    output_stems = {_stem(w) for w in _tokens(output_text)}
    flagged = sorted({
        _stem(term) for term in _CHARGED_TERMS
        if _stem(term) in output_stems and _stem(term) not in input_stems
    })
    return flagged


# ---------------------------------------------------------------- aggregate

def check(input_text: str, output_text: str) -> dict:
    """All conservation checks in one JSON-safe dict.

    Keys are stable API (the envelope's additive `conservation` field):
    unsupported_sentences, quote_violations, dropped_modifiers,
    format_flags, diction_flags.
    """
    return {
        "unsupported_sentences": unsupported_sentences(input_text, output_text),
        "quote_violations": quote_violations(input_text, output_text),
        "dropped_modifiers": dropped_modifiers(input_text, output_text),
        "format_flags": format_flags(input_text, output_text),
        "diction_flags": escalated_diction(input_text, output_text),
    }
