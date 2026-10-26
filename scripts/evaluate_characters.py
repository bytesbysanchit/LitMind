"""Character-extraction evaluation against a hand-annotated ground truth.

Ground truth = the author's own reading of two Project Gutenberg novels (single annotator, so treat
the numbers as indicative). Each real character has a set of surface names; known non-characters
(places, peoples, interjections) are listed separately.

Metrics per system output (a list of entries, each with a name and optional aliases):
  precision        entries that are real people / all entries
  core recall      core characters found / core characters
  duplicate        an entry that is the SAME person as a higher-ranked entry (fragmentation)
  wrong merge      one entry whose aliases belong to two different people
  unresolved       real-looking person entry whose identity cannot be fixed from the table (e.g. 'Mary')
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from litmind.characters import parse_surface

GT = {
 "dracula": {
  "core": ["dracula","jonathan","mina","lucy","vanhelsing","seward","arthur","quincey","renfield","hawkins"],
  "alias": {
    "dracula": ["dracula","count dracula","voivode dracula","voivode","the count"],
    "jonathan": ["jonathan harker","jonathan"], "mina": ["mina","mina murray","mina harker","murray","wilhelmina murray"],
    "lucy": ["lucy","lucy westenra","westenra"], "vanhelsing": ["van helsing","abraham van helsing","helsing","the professor"],
    "seward": ["seward","john seward","jack seward","john","jack"],
    "arthur": ["arthur","arthur holmwood","holmwood","art","godalming"],
    "quincey": ["quincey morris","quincey","morris","quincy"], "renfield": ["renfield"], "hawkins": ["hawkins","peter hawkins"]},
  "ambiguous": {"harker": "jonathan"},          # 'Harker' alone = Jonathan or Mina
  "invalid": ["czarina catherine","catherine","exeter","galatz","danube","bistritza","bistritz","god","wolf"]},
 "sherlock": {
  "core": ["holmes","watson","lestrade","adler","wilson","clay","ross","merryweather","jones","sutherland","windibank",
           "angel","mccarthy","turner","openshaw","stclair","baker","peterson","horner","stoner","roylott","hatherley",
           "stsimon","holder","hunter","rucastle","stark","burnwell","doran","bradstreet"],
  "alias": {
    "holmes": ["holmes","sherlock holmes","sherlock"], "watson": ["watson"], "lestrade": ["lestrade"],
    "adler": ["irene adler","adler","irene"], "wilson": ["jabez wilson","wilson"], "clay": ["john clay","clay"],
    "ross": ["duncan ross","ross"], "merryweather": ["merryweather"], "jones": ["peter jones","jones"],
    "sutherland": ["mary sutherland","sutherland"], "windibank": ["james windibank","windibank"],
    "angel": ["hosmer angel","angel","hosmer"], "mccarthy": ["james mccarthy","mccarthy","mccarthys","jim"],
    "turner": ["turner"], "openshaw": ["john openshaw","openshaw"], "stclair": ["neville st clair","st clair","neville st"],
    "baker": ["henry baker","baker","henry bakers"], "peterson": ["peterson"], "horner": ["john horner","horner"],
    "stoner": ["helen stoner","stoner","miss stoner","helen"], "roylott": ["grimesby roylott","roylott"],
    "hatherley": ["victor hatherley","hatherley"], "stark": ["lysander stark","stark","lysander"],
    "stsimon": ["st simon","robert st simon","robert walsingham de vere st simon","simon","clara st simon"],
    "doran": ["hatty doran","doran"], "adoran": ["aloysius doran"], "holder": ["alexander holder","holder"],
    "aholder": ["arthur"], "burnwell": ["george burnwell","burnwell"], "hunter": ["violet hunter","hunter"],
    "rucastle": ["rucastle","jephro rucastle"], "arucastle": ["alice rucastle"], "bradstreet": ["bradstreet"],
    "millar": ["flora millar","millar","flora"], "toller": ["toller"], "fowler": ["fowler"], "stoper": ["stoper"],
    "oakshott": ["oakshott"], "frank": ["frank"]},
  "ambiguous": {},
  # bare first names that belong to 2+ different people across the stories (Mary Sutherland / Mary Holder /
  # Mary Jane; John Clay / Turner / Openshaw / Horner; James McCarthy / Windibank / Ryder)
  "shared_given": ["mary","john","james"],
  "unresolved_given_names": ["mary","alice","john","james","george","peter","robert","jack"],
  "invalid": ["majesty","briony lodge","stoke moran","cooee","horsham","lee","swandam lane","surrey","alpha","alpha inn"]}}

def norm(s):
    _, toks = parse_surface(s if any(c.isupper() for c in s) else s.title())   # raw baseline strings are lower-cased
    return " ".join(t.lower() for t in toks) if toks else s.casefold().strip()

def evaluate(book, entries):
    g = GT[book]
    lookup = {a: i for i, al in g["alias"].items() for a in al}
    lookup.update(g["ambiguous"]); invalid = set(g["invalid"]); given = set(g.get("unresolved_given_names", []))
    covered, E = set(), len(entries)
    V = I = D = W = U = 0; unknown = []
    for e in entries:                                   # entries are assumed ranked by mentions
        names = [e["name"]] + list(e.get("aliases", {}))
        keys = [norm(n) for n in names]
        ids = {lookup[k] for k in keys if k in lookup and k not in g["ambiguous"]}
        main = norm(e["name"])
        if main in invalid or (not ids and all(k in invalid for k in keys)):
            I += 1; continue
        if main in g["ambiguous"]: ids = ids or {g["ambiguous"][main]}
        if not ids:
            if main in given: V += 1; U += 1
            else: unknown.append(e["name"]); V += 1; U += 1
            continue
        V += 1
        if len(ids) > 1 or any(k in g.get("shared_given", []) for k in keys[1:]): W += 1
        primary = lookup.get(main, next(iter(ids)))
        if primary in covered: D += 1
        covered.add(primary)
    core = set(g["core"]); hit = len(core & covered)
    uniq = V - D                                       # distinct-looking real people reported
    up = uniq / E if E else 0; rc = hit / len(core)
    return {"entries": E, "valid": V, "invalid": I, "precision": round(V/E, 3) if E else 0,
            "unique_precision": round(up, 3), "f1": round(2*up*rc/(up+rc), 3) if up+rc else 0,
            "core_found": hit, "core_total": len(core), "core_recall": round(hit/len(core), 3),
            "duplicates": D, "wrong_merges": W, "unresolved": U, "unknown": unknown}

if __name__ == "__main__":
    d = json.load(open("/tmp/eval_entries.json")); v1 = json.load(open("/tmp/v1_results.json"))
    res = {}
    for book in ("dracula", "sherlock"):
        res[book] = {}
        raw = [{"name": x["name"], "mentions": x["mentions"]} for x in d[book]["raw"]]
        res[book]["raw_ner"] = evaluate(book, raw)
        res[book]["v1_litmind"] = evaluate(book, [{"name": n, "mentions": c} for n, c in sorted(v1[book]["counts"].items(), key=lambda kv: -kv[1])])
        for cname, cv in d[book]["configs"].items():
            res[book][cname] = evaluate(book, cv["entries"])
        res[book]["_meta"] = {"min_mentions": d[book]["min_mentions"], "chars": d[book]["chars"], "sentences": d[book]["sentences"],
            "raw_distinct": d[book]["raw_distinct_person_strings"], "raw_mentions": d[book]["raw_person_mentions"],
            "secs": {k: v["secs"] for k, v in d[book]["configs"].items()}, "v1_secs": v1[book]["secs"]}
    json.dump(res, open("/home/claude/LitMind2/reports/character_eval.json", "w"), indent=1)
    for book in res:
        print("\n==", book)
        print(f"{'system':16s} {'E':>3s} {'valid':>5s} {'inv':>3s} {'prec':>5s} {'uprec':>5s} {'core':>6s} {'rec':>5s} {'dup':>3s} {'wm':>2s} {'unres':>5s}")
        for k, r in res[book].items():
            if k == "_meta": continue
            print(f"{k:16s} {r['entries']:3d} {r['valid']:5d} {r['invalid']:3d} {r['precision']:5.3f} {r['unique_precision']:5.3f} {r['core_found']:2d}/{r['core_total']:<3d} {r['core_recall']:5.3f} {r['duplicates']:3d} {r['wrong_merges']:2d} {r['unresolved']:5d}  f1={r['f1']:.3f}" + (f"   UNKNOWN:{r['unknown']}" if r['unknown'] else ""))
