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
    otherwise exact; a span quoted N times in the input must appear at
    least N times in the output, so dropping one of two identical
    quotes is still a violation.
    """
    output_normalized = _normalize_quotes(output_text)
    violations = []
    counted: dict[str, int] = {}
    for span in quoted_spans(input_text):
        counted[span] = counted.get(span, 0) + 1
        if output_normalized.count(span) < counted[span]:
            violations.append(span)
    return violations


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

# Word <-> digit equivalents so a rewrite of "fifty" to "50" (or back)
# still counts as the numeral surviving; without this the modifier
# check would silently skip and miss a dropped hedge.
_WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19",
    "twenty": "20", "thirty": "30", "forty": "40", "fifty": "50",
    "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
    "hundred": "100", "thousand": "1000", "million": "1000000",
    "billion": "1000000000", "dozen": "12",
}
# First mapping wins on digit collisions ("twelve" and "dozen" both
# map to 12; the reverse lookup prefers the plain number word).
_DIGIT_TO_WORD: dict[str, str] = {}
for _word, _digit in _WORD_TO_DIGIT.items():
    _DIGIT_TO_WORD.setdefault(_digit, _word)

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

    def _numeral_survived(token: str) -> bool:
        """The numeral, or a word/digit equivalent of it, is in the
        output ("fifty" rewritten to "50" still anchors its hedge)."""
        bare = token.lower().strip("'’-")
        if _stem(bare) in output_vocab or token.lower() in output_lower:
            return True
        alias = _WORD_TO_DIGIT.get(bare) or _DIGIT_TO_WORD.get(bare)
        return bool(alias) and bool(
            re.search(r"\b" + re.escape(alias) + r"\b", output_lower)
        )

    def _scan_sentence(input_tokens: list[str]) -> None:
        for index, token in enumerate(input_tokens):
            if not _is_numeric_token(token):
                continue
            # The numeral itself must survive for a modifier check to
            # make sense; a lost numeral is the number-checker's finding.
            if not _numeral_survived(token):
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
# "deadly" is deliberately absent: its stem collides with plain "dead"
# ("the line went dead") and "lethal" covers the register.
_CHARGED_TERMS = (
    "hunt", "hunting", "hunted",
    "war", "warfare", "weapon", "weaponize", "weaponized",
    "lethal",
    "destroy", "destroyed", "destroying",
    "crush", "crushed", "crushing",
    "attack", "attacked", "attacking", "assault",
    "brutal", "savage", "vicious", "ruthless",
    "gossip",
    "slaughter", "annihilate",
)


def _charged_key(word: str) -> str:
    """Family key for charged-term matching: the crude stem, minus a
    trailing "e" so "-ed"/"-ing" strips of -e verbs still meet their
    base form ("annihilated" -> "annihilat" == "annihilate")."""
    stem = _stem(word)
    return stem[:-1] if stem.endswith("e") else stem


_CHARGED_KEYS = {_charged_key(term) for term in _CHARGED_TERMS}


def escalated_diction(input_text: str, output_text: str) -> list[str]:
    """Charged terms present in the output but absent from the input.

    Matching is by stem family so inflections ("slaughtering",
    "annihilated") are caught; the flagged entry is the actual output
    word, never a lossy stem. The input side suppresses by the same
    family, so "hunted" in the input covers "hunting" in the output.
    """
    input_keys = {_charged_key(w) for w in _tokens(input_text)}
    return sorted({
        word for word in _tokens(output_text)
        if _charged_key(word) in _CHARGED_KEYS
        and _charged_key(word) not in input_keys
    })


# ------------------------------------------------------- named entities

# Capitalized words that are not names: months, weekdays, and the
# pronoun/sentence-furniture set the sentence-case rule cannot catch.
_ENTITY_SKIP = frozenset(
    """
    january february march april may june july august september october
    november december monday tuesday wednesday thursday friday saturday
    sunday i i'm i've i'll i'd the a an and but or so yet no not
    """.split()
)

_CAPITALIZED = re.compile(r"^[A-Z][A-Za-z&.'-]*$")


def named_entities(text: str) -> list[str]:
    """Heuristic named entities: runs of capitalized words.

    Deliberately dependency-free (the core is stdlib-only and
    offline-deterministic), so this is sentence-case NER, not a model:
    a run counts when it sits mid-sentence, spans two or more words,
    is an acronym, or (for a sentence-initial single word) also
    appears as an entity mid-sentence elsewhere in the text, so
    "Scientology pursued..." still resolves once Scientology shows up
    anywhere else. Lowercase particles inside names ("bin", "van")
    split the run; months, weekdays, and capitalized sentence
    furniture are skipped. Good enough to anchor a diction flag, not a
    general-purpose extractor.
    """
    runs: list[tuple[str, bool]] = []  # (name, is_initial_single_word)
    for sentence in split_sentences(text):
        tokens = [t for t in _TOKEN_SPLIT.split(sentence) if t]
        run: list[str] = []
        run_start = 0
        for index, token in enumerate(tokens + [""]):
            word = token.strip(".,")
            if word and _CAPITALIZED.match(word) and \
                    word.lower() not in _ENTITY_SKIP:
                if not run:
                    run_start = index
                run.append(word)
                continue
            if run:
                initial_single = run_start == 0 and len(run) == 1
                runs.append((" ".join(run), initial_single))
                run = []

    anchored = {
        name.lower() for name, initial_single in runs
        if not initial_single or name.isupper()
    }
    entities: list[str] = []
    seen: set[str] = set()
    for name, initial_single in runs:
        if initial_single and not name.isupper() and \
                name.lower() not in anchored:
            continue
        if name.lower() not in seen:
            seen.add(name.lower())
            entities.append(name)
    return entities


def diction_escalations(input_text: str, output_text: str) -> list[dict]:
    """Escalated charged terms aimed at named third parties.

    The field report's class 3 receipt ("pursuing its critics with
    legal tools" -> "hunting its critics", on a litigation-adjacent
    Scientology passage) is dangerous precisely because a named party
    is in range. Every escalated_diction term that shares a sentence
    with a heuristic named entity is reported with that context.
    Advisory, like the plain diction flags.
    """
    terms = escalated_diction(input_text, output_text)
    if not terms:
        return []
    term_keys = {_charged_key(t) for t in terms}
    # Entities resolve over the whole output so a sentence-initial name
    # anchored elsewhere still counts in the sentence being judged.
    all_entities = named_entities(output_text)
    escalations = []
    for sentence in split_sentences(output_text):
        # Report the sentence's own surface form: with two inflections
        # of one family in different sentences, each sentence must name
        # the word it actually contains.
        hit_terms = sorted({
            w for w in _tokens(sentence) if _charged_key(w) in term_keys
        })
        if not hit_terms:
            continue
        sentence_lower = sentence.lower()
        entities = [
            name for name in all_entities
            if re.search(
                r"\b" + re.escape(name.lower()) + r"\b", sentence_lower
            )
        ]
        if not entities:
            continue
        for term in hit_terms:
            escalations.append({
                "term": term,
                "entities": entities,
                "sentence": sentence,
            })
    return escalations


# ---------------------------------------------------------------- aggregate

def check(input_text: str, output_text: str) -> dict:
    """All conservation checks in one JSON-safe dict.

    Keys are stable API (the envelope's additive `conservation` field):
    unsupported_sentences, quote_violations, dropped_modifiers,
    format_flags, diction_flags, diction_escalations.
    """
    return {
        "unsupported_sentences": unsupported_sentences(input_text, output_text),
        "quote_violations": quote_violations(input_text, output_text),
        "dropped_modifiers": dropped_modifiers(input_text, output_text),
        "format_flags": format_flags(input_text, output_text),
        "diction_flags": escalated_diction(input_text, output_text),
        "diction_escalations": diction_escalations(input_text, output_text),
    }
