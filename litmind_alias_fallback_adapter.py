"""LitMind all-character alias fallback adapter v2.

Safe, separate diagnostic/candidate-recovery layer. It does NOT import or edit
src/nlp_processor.py or prev.py, and never treats a raw alias match as verified truth.

Input workbook sheets: CHARACTER_MASTER, UTTERANCE_CORPUS
Output: Excel report + candidate CSV.
Requires pandas, openpyxl, spaCy, and the chosen spaCy model in the current environment.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher

STAGE_CUES = re.compile(r"\b(turns? to|slowly comes|walks?|enters?|exits?|sits?|stands?|lies down|looks? at|moves? to|crosses?|dances?|pause|silence|blackout|lights? fade|stage direction)\b", re.I)
META_CUES = re.compile(r"\b(title|copyright|written by|author|novel|manuscript|book|playwright|publisher|published|address)\b", re.I)


def txt(v):
    return "" if pd.isna(v) else str(v).strip()


def classify_candidate(text: str, start: int, end: int, has_multiple_chars: bool):
    """Conservative heuristic flags; all matches still require review."""
    left, right = max(0, start - 100), min(len(text), end + 100)
    window = text[left:right]
    prefix = text[max(0, start - 180):start]
    # Is the mention inside round/square brackets, based on nearest unmatched opener.
    last_open = max(text.rfind("(", 0, start), text.rfind("[", 0, start))
    last_close = max(text.rfind(")", 0, start), text.rfind("]", 0, start))
    in_brackets = last_open > last_close
    meta = bool(META_CUES.search(window))
    stage = in_brackets or bool(STAGE_CUES.search(window))
    if meta:
        tier = "LOW_PRIORITY_METADATA_OR_TITLE_REVIEW"
    elif stage:
        tier = "STAGE_DIRECTION_OR_MIXED_REVIEW"
    elif has_multiple_chars:
        tier = "MULTI_CHARACTER_DIALOGUE_REVIEW"
    else:
        tier = "DIALOGUE_OR_UNCLEAR_REVIEW"
    flags=[]
    if in_brackets: flags.append("MENTION_INSIDE_BRACKETS")
    if STAGE_CUES.search(window): flags.append("STAGE_CUE_NEAR_MENTION")
    if meta: flags.append("POSSIBLE_TITLE_OR_METADATA_CONTEXT")
    if has_multiple_chars: flags.append("MULTIPLE_CANONICAL_NAMES_IN_UTTERANCE")
    if not flags: flags.append("NO_SIMPLE_HEURISTIC_FLAG")
    return tier, ";".join(flags), window


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input', default='Dattani_Tara_Corpus_Final.xlsx')
    ap.add_argument('--output', default='reports/character_alias_fallback_v2.xlsx')
    ap.add_argument('--model', default='en_core_web_sm')
    args=ap.parse_args()
    inp=Path(args.input); out=Path(args.output)
    if not inp.exists(): raise FileNotFoundError(f'Workbook not found: {inp.resolve()}')
    chars=pd.read_excel(inp, sheet_name='CHARACTER_MASTER')
    utt=pd.read_excel(inp, sheet_name='UTTERANCE_CORPUS')
    reqc={'Character_ID','Play_ID','Character_Name'}
    requ={'Speech_ID','Play_ID','Play','Speech_OCR','QA_Flag'}
    if not reqc.issubset(chars.columns): raise ValueError(f'CHARACTER_MASTER missing columns: {reqc-set(chars.columns)}')
    if not requ.issubset(utt.columns): raise ValueError(f'UTTERANCE_CORPUS missing columns: {requ-set(utt.columns)}')
    if utt['Speech_ID'].duplicated().any():
        duplicate_ids=utt.loc[utt['Speech_ID'].duplicated(keep=False),'Speech_ID'].astype(str).tolist()
    else: duplicate_ids=[]
    nlp=spacy.load(args.model)
    matcher=PhraseMatcher(nlp.vocab, attr='LOWER')
    lookup={}; valid=[]
    for _,r in chars.iterrows():
        name=txt(r['Character_Name']); pid=txt(r['Play_ID']); cid=txt(r['Character_ID'])
        if not name: continue
        key=f'LITMIND_ALIAS_{pid}_{cid}'
        matcher.add(key,[nlp.make_doc(name)])
        lookup[nlp.vocab.strings[key]]={'Character_ID':cid,'Play_ID':pid,'Character':name}
        valid.append({'Character_ID':cid,'Play_ID':pid,'Character':name})
    docs=nlp.pipe([txt(x) for x in utt['Speech_OCR']],batch_size=32)
    rows=[]
    for (_,u),doc in zip(utt.iterrows(),docs):
        pid=txt(u['Play_ID']); text=doc.text
        all_mentions=[]
        for mid,s,e in matcher(doc):
            c=lookup[mid]
            if c['Play_ID'] != pid: continue
            all_mentions.append((c,s,e))
        unique_char_count=len({x[0]['Character_ID'] for x in all_mentions})
        ents=list(doc.ents); persons=[e for e in ents if e.label_=='PERSON']
        for c,s,e in all_mentions:
            span=doc[s:e]
            exact=any(x.start==s and x.end==e and x.label_=='PERSON' for x in ents)
            overlap=any(x.start<e and x.end>s and x.label_=='PERSON' for x in ents)
            tier,flags,window=classify_candidate(text,span.start_char,span.end_char,unique_char_count>1)
            rows.append({
                'Speech_ID':txt(u['Speech_ID']),'Play_ID':pid,'Play':txt(u['Play']),
                'Character_ID':c['Character_ID'],'Character_Candidate':c['Character'],
                'Matched_Surface':span.text,'Start_Char':span.start_char,'End_Char':span.end_char,
                'Baseline_NER_Entities':'; '.join(f'{x.text}:{x.label_}' for x in ents),
                'Baseline_Exact_PERSON':bool(exact),'Baseline_PERSON_Overlap':bool(overlap),
                'Alias_Fallback_Candidate':True,'Candidate_Not_Exact_PERSON':not exact,
                'Review_Priority':tier,'Safeguard_Flags':flags,'Local_Context_Window':window,
                'QA_Flag':txt(u.get('QA_Flag','')),
                'Decision': 'REVIEW_REQUIRED — DO NOT AUTO-ACCEPT',
            })
    detail=pd.DataFrame(rows)
    if detail.empty:
        detail=pd.DataFrame(columns=['Speech_ID','Play_ID','Play','Character_ID','Character_Candidate','Matched_Surface','Start_Char','End_Char','Baseline_NER_Entities','Baseline_Exact_PERSON','Baseline_PERSON_Overlap','Alias_Fallback_Candidate','Candidate_Not_Exact_PERSON','Review_Priority','Safeguard_Flags','Local_Context_Window','QA_Flag','Decision'])
    summary=[]
    for c in valid:
        sub=detail[(detail.Character_ID==c['Character_ID'])&(detail.Play_ID==c['Play_ID'])]
        summary.append({'Play_ID':c['Play_ID'],'Character_ID':c['Character_ID'],'Character':c['Character'],
            'Alias_Candidate_Occurrences':len(sub),'Unique_Utterances':sub.Speech_ID.nunique() if len(sub) else 0,
            'Baseline_Exact_PERSON_Matches':int(sub.Baseline_Exact_PERSON.sum()) if len(sub) else 0,
            'Alias_Only_Candidates':int(sub.Candidate_Not_Exact_PERSON.sum()) if len(sub) else 0,
            'Review_Required_Count':len(sub),'Interpretation':'Diagnostic counts only; not accuracy or verified mentions.'})
    char_summary=pd.DataFrame(summary)
    priority=(detail.groupby(['Play_ID','Review_Priority'],dropna=False).agg(Candidate_Occurrences=('Speech_ID','size'),Unique_Utterances=('Speech_ID','nunique')).reset_index() if len(detail) else pd.DataFrame(columns=['Play_ID','Review_Priority','Candidate_Occurrences','Unique_Utterances']))
    context=(detail.groupby(['Play_ID','Safeguard_Flags'],dropna=False).size().reset_index(name='Candidate_Occurrences') if len(detail) else pd.DataFrame(columns=['Play_ID','Safeguard_Flags','Candidate_Occurrences']))
    overview=pd.DataFrame([
        ('Utterance rows processed',len(utt)),('Characters in master',len(valid)),('Alias occurrence candidates',len(detail)),
        ('Unique utterances with candidates',int(detail.Speech_ID.nunique()) if len(detail) else 0),
        ('Baseline exact PERSON matches among alias occurrences',int(detail.Baseline_Exact_PERSON.sum()) if len(detail) else 0),
        ('Alias-only candidates',int(detail.Candidate_Not_Exact_PERSON.sum()) if len(detail) else 0),
        ('Candidates requiring review',len(detail)),('Duplicate Speech_ID rows in corpus',len(duplicate_ids)),
        ('NER model',args.model),('spaCy version',spacy.__version__),
        ('Metric restriction','No accuracy/precision/recall/F1: no independent mention-level gold set.'),
        ('Safety policy','Alias matches are candidates only; no candidate is auto-accepted.'),
        ('Protected files','src/nlp_processor.py and prev.py are not imported or modified.'),
    ],columns=['Metric','Value'])
    methodology=pd.DataFrame([
        ('Alias matching','Case-insensitive PhraseMatcher token-sequence matching of canonical CHARACTER_MASTER names, scoped by Play_ID.'),
        ('Substring safeguard','Token-sequence matching avoids substring matches such as Dan inside Chandan.'),
        ('Baseline','Fresh spaCy entities from the selected model; exact and overlapping PERSON spans reported separately.'),
        ('Review safeguards','Bracket/stage cues, title/metadata cues, and multiple character names are flagged for review.'),
        ('No auto-accept','Every alias match remains REVIEW_REQUIRED. Heuristics are triage flags, not truth labels.'),
        ('Limitations','Canonical names only; nicknames/variants need a reviewed alias dictionary. OCR errors may evade exact matching.'),
        ('Evaluation','Independent verified gold mentions are required before precision/recall/F1/accuracy claims.'),
    ],columns=['Item','Detail'])
    out.parent.mkdir(parents=True,exist_ok=True)
    csvout=out.with_name(out.stem+'_candidates.csv')
    with pd.ExcelWriter(out,engine='openpyxl') as w:
        overview.to_excel(w,sheet_name='Summary',index=False)
        char_summary.to_excel(w,sheet_name='Character_Summary',index=False)
        priority.to_excel(w,sheet_name='Review_Priority',index=False)
        context.to_excel(w,sheet_name='Safeguard_Flags',index=False)
        detail.to_excel(w,sheet_name='Candidate_Review_Queue',index=False)
        methodology.to_excel(w,sheet_name='Methodology',index=False)
    detail.to_csv(csvout,index=False,encoding='utf-8-sig')
    print('LITMIND ALIAS FALLBACK ADAPTER V2 COMPLETE')
    print(f'Utterances processed: {len(utt)}')
    print(f'Characters: {len(valid)}')
    print(f'Alias occurrence candidates: {len(detail)}')
    print(f'Baseline exact PERSON matches: {int(detail.Baseline_Exact_PERSON.sum()) if len(detail) else 0}')
    print(f'Alias-only candidates: {int(detail.Candidate_Not_Exact_PERSON.sum()) if len(detail) else 0}')
    print(f'Excel report: {out.resolve()}')
    print(f'Candidate CSV: {csvout.resolve()}')
    print('All matches remain review candidates; diagnostic only, not an accuracy score.')

if __name__=='__main__': main()
