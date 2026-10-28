"""Character and alias extraction.

Same base method as the old LitMind (spaCy NER) but with the three fixes that
address the reported problems:

1. LOCATIONS COUNTED AS CHARACTERS
   - Label voting: a name is only kept if spaCy tags it PERSON more often than
     GPE/LOC/FAC/ORG/NORP/... across the whole book.
   - Dependency evidence: names that mostly appear as the object of a locative
     preposition ("in Whitby", "near Exeter") and never as a subject are places.
   - Place-word filter ("Baker Street", "Castle Dracula") and common-word filter
     (words that also appear in lower case are not names).

2. ONE PERSON COUNTED AS MANY CHARACTERS
   - Titles are stripped ("Mr. Holmes" == "Holmes" == "Sherlock Holmes").
   - Short names are attached to the longer name that contains them
     ("Mina" -> "Mina Harker"). When a short name fits several people
     ("Harker" -> Jonathan / Mina) it is resolved by title/pronoun gender and by
     dominance, and left unassigned (reported in `rejected`) if still ambiguous.
   - Near-identical spellings are merged ("Quincey"/"Quincy").
   - Optional: a Gemini review pass can merge aliases that need world knowledge
     ("the Count" = "Dracula"); see `apply_review`.

Everything here is deterministic and works offline.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
MALE_TITLES = {"mr", "mister", "sir", "lord", "monsieur", "herr", "master", "king", "prince",
               "duke", "baron", "count", "uncle", "brother", "signor", "don", "emperor", "czar",
               "tsar", "pope", "friar", "abbot"}
FEMALE_TITLES = {"mrs", "miss", "ms", "lady", "madam", "madame", "mme", "mademoiselle", "mlle",
                 "queen", "princess", "duchess", "baroness", "countess", "aunt", "sister",
                 "frau", "signora", "dame", "empress", "czarina", "tsarina"}
NEUTRAL_TITLES = {"dr", "doctor", "professor", "prof", "captain", "capt", "colonel", "col", "major",
                  "general", "inspector", "detective", "father", "mother", "reverend", "rev",
                  "judge", "sergeant", "sgt", "lieutenant", "lt", "admiral", "commissioner",
                  "constable", "squire", "saint", "bishop", "archbishop", "cardinal"}
TITLES = MALE_TITLES | FEMALE_TITLES | NEUTRAL_TITLES

PARTICLES = {"van", "von", "de", "der", "den", "la", "le", "du", "di", "da", "del", "della",
             "al", "bin", "ibn", "el"}

# Words that end up glued to a name when a heading like "DR. SEWARD'S DIARY" is tagged as PERSON.
NOISE_SUFFIX = {"diary", "journal", "memorandum", "memoranda", "mem", "letter", "telegram", "note",
                "notes", "phonograph", "narrative", "statement", "chapter"}

# Common English nicknames -> formal names (used only to merge two names that BOTH occur).
NICKNAMES = {
    "art": {"arthur"}, "jack": {"john"}, "johnny": {"john"}, "bill": {"william"}, "billy": {"william"}, "will": {"william"},
    "willie": {"william"}, "tom": {"thomas"}, "tommy": {"thomas"}, "jim": {"james"},
    "jimmy": {"james"}, "bob": {"robert"}, "bobby": {"robert"}, "rob": {"robert"},
    "dick": {"richard"}, "rick": {"richard"}, "harry": {"henry", "harold"}, "hal": {"henry"},
    "ned": {"edward", "edmund"}, "ted": {"edward", "theodore"}, "eddie": {"edward"},
    "mike": {"michael"}, "dave": {"david"}, "dan": {"daniel"}, "danny": {"daniel"},
    "sam": {"samuel"}, "ben": {"benjamin"}, "joe": {"joseph"}, "charlie": {"charles"},
    "chuck": {"charles"}, "fred": {"frederick", "alfred"}, "freddie": {"frederick"},
    "alf": {"alfred"}, "andy": {"andrew"}, "tony": {"anthony"}, "nick": {"nicholas"},
    "pete": {"peter"}, "steve": {"stephen", "steven"}, "walt": {"walter"}, "wally": {"walter"},
    "kate": {"katherine", "catherine", "kathleen"}, "katie": {"katherine", "catherine"},
    "kitty": {"katherine", "catherine"}, "liz": {"elizabeth"}, "lizzie": {"elizabeth"},
    "beth": {"elizabeth"}, "betty": {"elizabeth"}, "eliza": {"elizabeth"}, "maggie": {"margaret"},
    "meg": {"margaret"}, "peggy": {"margaret"}, "molly": {"mary"}, "polly": {"mary"},
    "nell": {"eleanor", "helen"}, "nelly": {"eleanor", "helen"}, "sally": {"sarah"},
    "sue": {"susan"}, "susie": {"susan"}, "jenny": {"jennifer", "jane"}, "jess": {"jessica"},
    "annie": {"anne", "anna"}, "fanny": {"frances"}, "fan": {"frances"}, "patty": {"patricia"},
    "mina": {"wilhelmina"}, "lucy": {"lucia"},
}

PLACE_WORDS = {
    "castle", "hotel", "street", "road", "abbey", "hall", "house", "church", "lane", "square",
    "park", "station", "bridge", "college", "university", "hospital", "river", "sea", "mount",
    "mountain", "lake", "island", "gate", "tower", "inn", "club", "court", "avenue", "bay",
    "port", "harbour", "harbor", "village", "town", "city", "county", "forest", "valley",
    "pass", "cathedral", "palace", "museum", "bank", "office", "market", "garden", "gardens",
    "cottage", "farm", "manor", "monastery", "convent", "asylum", "prison", "theatre", "theater",
    "cemetery", "churchyard", "cross", "circus", "terrace", "crescent", "row", "place", "end",
}

# Not people, even when spaCy says PERSON.
NON_NAMES = {
    "god", "heaven", "hell", "lord", "sir", "madam", "madame", "christ", "jesus", "satan", "devil",
    "christmas", "easter", "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday", "chapter", "page", "volume", "mister", "miss", "mrs", "mr",
    "english", "french", "german", "latin", "greek", "hungarian", "london", "paris",
    "majesty", "highness", "excellency", "lordship", "ladyship", "worship", "honour", "grace",
}

LOCATIVE_PREPS = {"in", "near", "into", "through", "across", "towards", "toward", "outside",
                  "inside", "around", "throughout", "within", "beyond", "to", "on", "down",
                  "along", "up", "over"}
# "to" is only counted as locative when the governing verb is a motion verb:
MOTION_VERBS = {"go", "come", "travel", "journey", "drive", "ride", "return", "arrive", "walk",
                "fly", "sail", "move", "proceed", "hurry", "run", "depart", "leave", "reach", "get"}

VOTE_LABELS = {"PERSON", "GPE", "LOC", "FAC", "ORG", "NORP", "EVENT", "LAW", "LANGUAGE",
               "PRODUCT", "WORK_OF_ART"}

MALE_PRON = {"he", "him", "his", "himself"}
FEMALE_PRON = {"she", "her", "hers", "herself"}

_TITLE_RE = re.compile(
    r"\b(?:Mr|Mrs|Miss|Ms|Dr|Professor|Prof|Sir|Lady|Lord|Captain|Capt|Colonel|Col|Major|Inspector|"
    r"Madame|Mme|Monsieur|Mademoiselle|Mlle|Reverend|Rev|Count|Countess|Baron|Duke|Duchess|Uncle|"
    r"Aunt|Sergeant)\b\.?\s+"
    r"((?:(?:Van|Von|De|La|Le|Du|Di|Der)\s+[A-Z][\w'’-]+)|(?:[A-Z][\w'’-]+))"
)
_BEFORE_TITLE = re.compile(
    r"\b(" + "|".join(sorted(TITLES, key=len, reverse=True)) + r")\.?\s+$", re.I)
_THE_TITLE = re.compile(
    r"\b[Tt]he\s+(Count|Countess|Professor|Captain|Colonel|Major|Inspector|Baron|Duke|Duchess|"
    r"Czarina|Emperor|Empress|Bishop|Judge|Sergeant|Constable|Squire)\b(?!\s+[A-Z])")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_TOKEN_OK = re.compile(r"[A-Za-z][A-Za-z'\-]*")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Character:
    name: str
    aliases: dict[str, int]            # alias (title-free) -> mention count
    mentions: int
    gender: str | None = None          # "M", "F" or None
    titles: list[str] = field(default_factory=list)
    sentence_ids: list[int] = field(default_factory=list)
    note: str = ""                     # e.g. "short name also fits: Mina Murray, Mina Harker"

    @property
    def alias_list(self) -> list[str]:
        return [a for a, _ in sorted(self.aliases.items(), key=lambda kv: -kv[1]) if a != self.name]


@dataclass
class CharacterResult:
    characters: list[Character]
    rejected: list[dict]               # {"name", "mentions", "reason"}
    sentences: list[str]
    n_chars: int

    def by_name(self, name: str) -> Character | None:
        low = name.casefold()
        for c in self.characters:
            if c.name.casefold() == low or low in (a.casefold() for a in c.aliases):
                return c
        return None


@dataclass
class _Mention:
    start: int
    end: int
    sent: int
    label: str
    surface: str
    cat: str                           # "subj" | "place" | "other"


@dataclass
class _Cand:
    key: str
    tokens: tuple[str, ...]
    count: int = 0
    titles: Counter = field(default_factory=Counter)
    subj: int = 0
    place: int = 0
    male: int = 0
    female: int = 0
    sents: set = field(default_factory=set)

    def gender(self) -> str | None:
        m = sum(v for t, v in self.titles.items() if t in MALE_TITLES)
        f = sum(v for t, v in self.titles.items() if t in FEMALE_TITLES)
        if m != f:
            return "M" if m > f else "F"
        tot = self.male + self.female
        if tot >= 5:
            if self.male / tot >= 0.7:
                return "M"
            if self.female / tot >= 0.7:
                return "F"
        return None


# --------------------------------------------------------------------------- #
# Surface-form parsing
# --------------------------------------------------------------------------- #
def parse_surface(surface: str) -> tuple[str | None, tuple[str, ...]]:
    """'Dr. John Seward's' -> ('dr', ('John', 'Seward'));  'the Count' -> ('count', ())."""
    s = surface.replace("’", "'").strip()
    s = re.sub(r"'s\b", "", s, flags=re.I)
    s = re.sub(r"^[^A-Za-z]+|[^A-Za-z.]+$", "", s)
    toks = s.split()
    title = None
    while toks and (toks[0].rstrip(".").lower() in TITLES or toks[0].lower() == "the"):
        t = toks.pop(0).rstrip(".").lower()
        if t != "the" and title is None:
            title = t
    kept: list[str] = []
    for t in toks:
        base = t.rstrip(".")
        if not base:
            continue
        if base.lower() in PARTICLES and kept:
            kept.append(base.lower())
            continue
        if not _TOKEN_OK.fullmatch(base) or not base[0].isupper():
            break
        if len(base) == 1:            # initials ("J.") carry no matching value
            continue
        if base.isupper() and len(base) > 1:
            base = base.capitalize()
        kept.append(base)
    while kept and (kept[-1] in PARTICLES or kept[-1].lower() in NOISE_SUFFIX):
        kept.pop()
    if len(kept) == 1 and kept[0].lower() in PARTICLES:
        kept = []
    return title, tuple(kept)


def _key(tokens: Iterable[str]) -> str:
    return " ".join(t.lower() for t in tokens)


# --------------------------------------------------------------------------- #
# NLP scan
# --------------------------------------------------------------------------- #
def load_nlp(model: str = "en_core_web_sm"):
    import spacy
    try:
        return spacy.load(model, disable=["lemmatizer"])
    except OSError as exc:  # pragma: no cover
        raise RuntimeError(
            f"spaCy model '{model}' is not installed. Run: python -m spacy download {model}"
        ) from exc


def _chunks(text: str, size: int = 120_000):
    pos, n = 0, len(text)
    while pos < n:
        end = min(pos + size, n)
        if end < n:
            cut = text.rfind("\n\n", pos, end)
            if cut > pos + size // 2:
                end = cut + 2
        yield pos, text[pos:end]
        pos = end


def _scan(text: str, nlp):
    sentences: list[str] = []
    sent_starts: list[int] = []
    male_pron: list[int] = []
    female_pron: list[int] = []
    mentions: list[_Mention] = []

    for base, chunk in _chunks(text):
        doc = nlp(chunk)
        sid_by_start: dict[int, int] = {}
        for sent in doc.sents:
            sid_by_start[sent.start_char] = len(sentences)
            sentences.append(sent.text.strip())
            sent_starts.append(base + sent.start_char)
            lows = [t.lower_ for t in sent]
            male_pron.append(sum(w in MALE_PRON for w in lows))
            female_pron.append(sum(w in FEMALE_PRON for w in lows))
        for ent in doc.ents:
            if ent.label_ not in VOTE_LABELS:
                continue
            root = ent.root
            cat = "other"
            if root.dep_ in ("nsubj", "nsubjpass"):
                cat = "subj"
            elif root.dep_ == "pobj" and root.head.lower_ in LOCATIVE_PREPS:
                head = root.head
                if head.lower_ != "to" or head.head.lemma_.lower() in MOTION_VERBS or \
                        head.head.text.lower() in MOTION_VERBS:
                    cat = "place"
            mentions.append(_Mention(base + ent.start_char, base + ent.end_char,
                                     sid_by_start[ent.sent.start_char], ent.label_, ent.text, cat))
        del doc
    return sentences, sent_starts, male_pron, female_pron, mentions


def _regex_recall(text: str, mentions: list[_Mention], sent_starts: list[int]) -> list[_Mention]:
    """Add 'Mr. Holmes'-style mentions that NER missed."""
    person = sorted((m.start, m.end) for m in mentions if m.label == "PERSON")
    starts = [p[0] for p in person]
    extra: list[_Mention] = []
    for m in _TITLE_RE.finditer(text):
        i = bisect_right(starts, m.start()) - 1
        overlap = (i >= 0 and person[i][1] > m.start()) or \
                  (i + 1 < len(person) and person[i + 1][0] < m.end())
        if overlap:
            continue
        sid = max(0, bisect_right(sent_starts, m.start()) - 1)
        extra.append(_Mention(m.start(), m.end(), sid, "PERSON", m.group(0).strip(), "other"))
    return extra


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def _nick_equal(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    return a == b or b in NICKNAMES.get(a, ()) or a in NICKNAMES.get(b, ())


def _lower_ratio(word_counts: Counter, w: str) -> float:
    low = word_counts.get(w.lower(), 0)
    cap = word_counts.get(w.capitalize(), 0) + word_counts.get(w.upper(), 0)
    return low / (low + cap) if (low + cap) else 0.0


def extract_characters(
    text: str,
    nlp=None,
    *,
    min_mentions: int | None = None,
    max_characters: int = 40,
    model: str = "en_core_web_sm",
) -> CharacterResult:
    nlp = nlp or load_nlp(model)
    sentences, sent_starts, male_pron, female_pron, mentions = _scan(text, nlp)
    mentions += _regex_recall(text, mentions, sent_starts)
    if min_mentions is None:
        min_mentions = max(5, len(text) // 100_000)

    word_counts = Counter(_WORD_RE.findall(text))
    the_counts = Counter(re.findall(r"\b[Tt]he\s+([A-Z][a-z]+)\b", text))   # "the Danube", "the Turk"
    stop = set(nlp.Defaults.stop_words)
    rejected: list[dict] = []

    # ---- 1. label vote across every entity type ---------------------------
    vote: dict[str, Counter] = defaultdict(Counter)
    parsed: dict[int, tuple[str | None, tuple[str, ...]]] = {}
    for i, m in enumerate(mentions):
        title, toks = parse_surface(m.surface)
        toks = list(toks)
        while len(toks) > 1 and (toks[0].lower() in stop or _lower_ratio(word_counts, toks[0]) > 0.5):
            toks.pop(0)
        toks = tuple(toks)
        if title is None and m.label == "PERSON":       # "Mrs. [Harker]" -> title sits before the span
            before = _BEFORE_TITLE.search(text[max(0, m.start - 16): m.start])
            if before:
                title = before.group(1).lower()
        parsed[i] = (title, toks)
        if toks:
            vote[_key(toks)][m.label] += 1

    # ---- 2. build PERSON candidates --------------------------------------
    cands: dict[str, _Cand] = {}
    sent_people: dict[int, set] = defaultdict(set)
    for i, m in enumerate(mentions):
        if m.label != "PERSON":
            continue
        title, toks = parsed[i]
        if not toks:
            continue
        k = _key(toks)
        c = cands.setdefault(k, _Cand(k, toks))
        c.count += 1
        c.sents.add(m.sent)
        if title:
            c.titles[title] += 1
        if m.cat == "subj":
            c.subj += 1
        elif m.cat == "place":
            c.place += 1
        sent_people[m.sent].add(k)
    # pronoun gender evidence from sentences where exactly one person is mentioned
    for sid, keys in sent_people.items():
        if len(keys) != 1:
            continue
        k = next(iter(keys))
        for s_ in (sid, sid + 1):
            if s_ < len(sentences):
                cands[k].male += male_pron[s_]
                cands[k].female += female_pron[s_]

    def reject(c: _Cand, reason: str):
        rejected.append({"name": " ".join(c.tokens), "mentions": c.count, "reason": reason})

    # ---- 3. hard filters on single surface forms --------------------------
    for k in list(cands):
        c = cands[k]
        why = None
        if len(c.tokens) == 1 and len(c.tokens[0]) < 3:
            why = "too short"
        elif k in NON_NAMES or (len(c.tokens) == 1 and c.tokens[0].lower() in NON_NAMES):
            why = "not a person name"
        elif any(t.lower() in PLACE_WORDS for t in c.tokens) and not c.titles:
            why = "contains a place word"
        elif len(c.tokens) == 1:
            w = c.tokens[0]
            low = word_counts.get(w.lower(), 0)
            cap = word_counts.get(w.capitalize(), 0) + word_counts.get(w.upper(), 0)
            if w.lower() in stop:
                why = "common word"
            elif low and low / (low + cap) > 0.2:
                why = "also appears in lower case (common word)"
            elif not c.titles and cap and the_counts.get(w, 0) / cap >= 0.3:
                why = "usually written 'the X' (river, people, ship or thing)"
        if why is None and c.place >= 3 and c.place > 1.5 * (c.subj + sum(c.titles.values())) \
                and not c.titles:
            why = f"used as a location ({c.place}x after 'in/near/on…') and rarely as an actor"
        if why:
            reject(c, why)
            del cands[k]

    # ---- 4. cluster aliases (union-find) ---------------------------------
    parent = {k: k for k in cands}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):  # b becomes the root
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    def agg(root):
        members = [cands[k] for k in cands if find(k) == root]
        pseudo = _Cand(root, cands[root].tokens)
        for m in members:
            pseudo.titles.update(m.titles)
        pseudo.male = sum(m.male for m in members)
        pseudo.female = sum(m.female for m in members)
        pseudo.count = sum(m.count for m in members)
        return pseudo

    notes: dict[str, str] = {}
    # 4a. same-length nickname pairs: "Art"~"Arthur", "Jack Seward"~"John Seward".
    #     Only when the nickname matches exactly ONE other name (never chain people together).
    for a in sorted(cands, key=lambda x: -cands[x].count):
        ta = cands[a].tokens
        matches = [b for b in cands
                   if b != a and len(cands[b].tokens) == len(ta) and cands[b].tokens[1:] == ta[1:]
                   and ta[0].lower() in NICKNAMES and cands[b].tokens[0].lower() in NICKNAMES[ta[0].lower()]]
        if len(matches) == 1 and find(a) != find(matches[0]):
            ga, gb = agg(find(a)).gender(), agg(find(matches[0])).gender()
            if not (ga and gb and ga != gb):
                union(a, find(matches[0]))

    # 4b. short name -> the longer name that contains it
    for k in sorted(cands, key=lambda x: -len(cands[x].tokens)):
        c = cands[k]
        tset = set(t.lower() for t in c.tokens)
        parents: dict[str, None] = {}
        for k2, c2 in cands.items():
            if len(c2.tokens) > len(c.tokens) and tset <= set(t.lower() for t in c2.tokens):
                parents[find(k2)] = None
        if not parents or find(k) in parents:      # no parent, or already attached to one
            continue
        options = [agg(r) for r in parents]
        gx = agg(find(k)).gender()
        compat = [o for o in options if not (gx and o.gender() and o.gender() != gx)]
        compat.sort(key=lambda o: -o.count)
        if len(compat) == 1:
            union(k, compat[0].key)
        elif len(compat) > 1 and compat[0].count >= 1.5 * compat[1].count:
            union(k, compat[0].key)
            notes[compat[0].key] = (notes.get(compat[0].key, "") +
                                    f" '{' '.join(c.tokens)}' assigned here by frequency; also fits "
                                    f"{' '.join(compat[1].tokens)}.").strip()
        else:  # keep it as its own character instead of guessing
            names = ", ".join(" ".join(o.tokens) for o in (compat or options)[:3])
            notes[k] = f"short name also fits: {names}"

    # 4c. spelling variants ("Quincey"/"Quincy"): every token identical, or a near-identical long token
    def _same_spelling(x: str, y: str) -> bool:
        return x == y or (len(x) >= 5 and len(y) >= 5 and x[0] == y[0] and abs(len(x) - len(y)) <= 1
                          and SequenceMatcher(None, x, y).ratio() >= 0.9)

    keys = sorted(cands, key=lambda x: -cands[x].count)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ta, tb = a.split(), b.split()
            if find(a) != find(b) and len(ta) == len(tb) and a != b and \
                    all(_same_spelling(x, y) for x, y in zip(ta, tb)):
                union(b, find(a))

    # ---- 5. assemble characters ------------------------------------------
    groups: dict[str, list[str]] = defaultdict(list)
    for k in cands:
        groups[find(k)].append(k)

    chars: list[Character] = []
    for root, members in groups.items():
        total = sum(cands[m].count for m in members)
        label = " ".join(max((cands[m] for m in members), key=lambda c: c.count).tokens)
        person_votes = sum(vote[m]["PERSON"] for m in members)
        other_votes = sum(n for m in members for lab, n in vote[m].items() if lab != "PERSON")
        subj = sum(cands[m].subj for m in members)
        title_n = sum(sum(cands[m].titles.values()) for m in members)
        gender_known = agg(root).gender() is not None
        strong = total >= 5 and (gender_known or subj >= max(2, 0.1 * total) or title_n >= 3)
        why = None
        if total < min_mentions:
            why = f"fewer than {min_mentions} mentions"
        elif other_votes > person_votes and not strong:
            why = f"tagged as place/organisation {other_votes}x vs person {person_votes}x"
        elif total < 15 and not (title_n or subj >= 2 or gender_known):
            why = "no sign of acting as a person (rarely a subject, no title, no he/she)"
        if why:
            rejected.append({"name": label, "mentions": total, "reason": why})
            continue
        # display name: the fullest well-attested form, not a one-off glitch
        floor = max(2, 0.03 * total)
        solid = [cands[m] for m in members if cands[m].count >= floor and len(cands[m].tokens) >= 2]
        best = max(solid, key=lambda c: (c.count, len(c.tokens))) if solid else \
            max((cands[m] for m in members), key=lambda c: c.count)
        titles: Counter = Counter()
        sids: set = set()
        for m in members:
            titles.update(cands[m].titles)
            sids |= cands[m].sents
        pseudo = agg(root)
        note = next((notes[m] for m in members if m in notes and m == best.key), "")
        chars.append(Character(
            name=" ".join(best.tokens),
            aliases={" ".join(cands[m].tokens): cands[m].count for m in members},
            mentions=total,
            gender=pseudo.gender(),
            titles=[t for t, _ in titles.most_common(3)],
            sentence_ids=sorted(sids),
            note=note,
        ))

    # ---- 6. "the Count", "the Professor": attach to the ONLY holder of that title ----
    holders: dict[str, list[Character]] = defaultdict(list)
    for ch in chars:
        for t in ch.titles:
            holders[t].append(ch)
    for m in _THE_TITLE.finditer(text):
        t = m.group(1).lower()
        if len(holders.get(t, [])) == 1:
            ch = holders[t][0]
            label = "the " + m.group(1)
            ch.aliases[label] = ch.aliases.get(label, 0) + 1
            ch.mentions += 1
            sid = max(0, bisect_right(sent_starts, m.start()) - 1)
            if sid not in ch.sentence_ids:
                ch.sentence_ids.append(sid)
    for ch in chars:
        ch.sentence_ids.sort()

    chars.sort(key=lambda c: -c.mentions)
    rejected.sort(key=lambda r: -r["mentions"])
    return CharacterResult(chars[:max_characters], rejected, sentences, len(text))


# --------------------------------------------------------------------------- #
# Optional LLM review (names only; no text is changed)
# --------------------------------------------------------------------------- #
def apply_review(result: CharacterResult, merge_groups: list[list[str]], drop: list[str]) -> CharacterResult:
    """Apply a reviewer's decisions (e.g. from Gemini): merge aliases / drop non-characters.

    merge_groups: [["Dracula", "Count Dracula"], ...] – first name in each group is kept.
    drop: names that are not characters.
    Unknown names are ignored, so a hallucinated name can never create a character.
    """
    chars = list(result.characters)

    def find(name: str) -> Character | None:
        low = name.casefold().strip()
        for c in chars:
            if c.name.casefold() == low or low in (a.casefold() for a in c.aliases):
                return c
        return None

    for name in drop:
        c = find(name)
        if c is not None:
            chars.remove(c)
    for group in merge_groups:
        found = [c for n in group if (c := find(n)) is not None]
        uniq = list(dict.fromkeys(id(c) for c in found))
        if len(uniq) < 2:
            continue
        keep = found[0]
        for other in found[1:]:
            if other is keep or other not in chars:
                continue
            for a, n in other.aliases.items():
                keep.aliases[a] = keep.aliases.get(a, 0) + n
            keep.mentions += other.mentions
            keep.sentence_ids = sorted(set(keep.sentence_ids) | set(other.sentence_ids))
            keep.titles = list(dict.fromkeys(keep.titles + other.titles))[:3]
            chars.remove(other)
    chars.sort(key=lambda c: -c.mentions)
    return CharacterResult(chars, result.rejected, result.sentences, result.n_chars)
