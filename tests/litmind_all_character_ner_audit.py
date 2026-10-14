"""LitMind: compare canonical character-name alias matches with spaCy NER.

Reads CHARACTER_MASTER and UTTERANCE_CORPUS from the Dattani corpus workbook.
Does not edit the source workbook or import/modify LitMind's frozen NLP files.
Run inside the user's LitMind .venv where en_core_web_sm is installed.
"""
from pathlib import Path
import argparse
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='Dattani_Tara_Corpus_Final.xlsx')
    parser.add_argument('--output', default='reports/character_alias_ner_comparison.xlsx')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f'Corpus workbook not found: {input_path.resolve()}')

    characters = pd.read_excel(input_path, sheet_name='CHARACTER_MASTER')
    utterances = pd.read_excel(input_path, sheet_name='UTTERANCE_CORPUS')
    required_chars = {'Character_ID', 'Play_ID', 'Character_Name'}
    required_utt = {'Speech_ID', 'Play_ID', 'Play', 'Speech_OCR', 'QA_Flag'}
    if not required_chars.issubset(characters.columns):
        raise ValueError(f'CHARACTER_MASTER missing columns: {sorted(required_chars-set(characters.columns))}')
    if not required_utt.issubset(utterances.columns):
        raise ValueError(f'UTTERANCE_CORPUS missing columns: {sorted(required_utt-set(utterances.columns))}')

    nlp = spacy.load('en_core_web_sm')
    matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
    id_to_character = {}
    for _, char in characters.iterrows():
        name = str(char['Character_Name']).strip()
        if not name or name.lower() == 'nan':
            continue
        key = f"CHAR_{char['Play_ID']}_{char['Character_ID']}"
        matcher.add(key, [nlp.make_doc(name)])
        id_to_character[nlp.vocab.strings[key]] = {
            'Character_ID': char['Character_ID'],
            'Play_ID': char['Play_ID'],
            'Character_Name': name,
        }

    detail_rows = []
    texts = [str(x) if pd.notna(x) else '' for x in utterances['Speech_OCR']]
    for (_, utt), doc in zip(utterances.iterrows(), nlp.pipe(texts, batch_size=32)):
        for match_id, start, end in matcher(doc):
            char = id_to_character[match_id]
            if str(char['Play_ID']) != str(utt['Play_ID']):
                continue
            span = doc[start:end]
            overlaps_person = any(ent.label_ == 'PERSON' and ent.start < end and ent.end > start for ent in doc.ents)
            exact_person = any(ent.label_ == 'PERSON' and ent.start == start and ent.end == end for ent in doc.ents)
            detail_rows.append({
                'Speech_ID': utt['Speech_ID'],
                'Play_ID': utt['Play_ID'],
                'Play': utt['Play'],
                'Character_ID': char['Character_ID'],
                'Character': char['Character_Name'],
                'Matched_Surface': span.text,
                'NER_Entities': '; '.join(f'{ent.text}:{ent.label_}' for ent in doc.ents),
                'NER_Exact_PERSON': exact_person,
                'NER_PERSON_Overlap': overlaps_person,
                'QA_Flag': utt['QA_Flag'],
                'Text_Context': doc.text,
            })

    detail = pd.DataFrame(detail_rows)
    if detail.empty:
        detail = pd.DataFrame(columns=['Speech_ID','Play_ID','Play','Character_ID','Character','Matched_Surface','NER_Entities','NER_Exact_PERSON','NER_PERSON_Overlap','QA_Flag','Text_Context'])

    summary_rows = []
    for _, char in characters.iterrows():
        name = str(char['Character_Name']).strip()
        subset = detail[(detail['Character_ID'] == char['Character_ID']) & (detail['Play_ID'].astype(str) == str(char['Play_ID']))]
        summary_rows.append({
            'Play_ID': char['Play_ID'],
            'Character_ID': char['Character_ID'],
            'Character': name,
            'Canonical_alias_occurrences': len(subset),
            'Unique_utterances_with_alias': subset['Speech_ID'].nunique() if len(subset) else 0,
            'Exact_PERSON_matches': int(subset['NER_Exact_PERSON'].sum()) if len(subset) else 0,
            'PERSON_overlap_matches': int(subset['NER_PERSON_Overlap'].sum()) if len(subset) else 0,
            'Alias_hits_without_exact_PERSON': int((~subset['NER_Exact_PERSON']).sum()) if len(subset) else 0,
            'Interpretation': 'Diagnostic only; canonical-name match is not proof of a true character reference.'
        })
    summary = pd.DataFrame(summary_rows)
    notes = pd.DataFrame([
        {'Item':'Purpose','Detail':'Compare exact canonical character-name matches against fresh spaCy en_core_web_sm PERSON entities.'},
        {'Item':'Scope','Detail':'Uses only canonical Character_Name values; nicknames/variants are not included unless present as canonical names.'},
        {'Item':'No accuracy claim','Detail':'Counts are diagnostic coverage/comparison, not precision, recall, F1, or overall accuracy. No independent gold labels are supplied.'},
        {'Item':'Stage directions','Detail':'Speech_OCR can contain dialogue and stage directions; alias matches in either context are included and require contextual interpretation.'},
        {'Item':'Safety','Detail':'Source workbook and frozen nlp_processor.py / prev.py are not modified.'},
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        summary.to_excel(writer, sheet_name='Character_Summary', index=False)
        detail.to_excel(writer, sheet_name='Alias_Mention_Details', index=False)
        notes.to_excel(writer, sheet_name='Methodology', index=False)
    print('LITMIND ALL-CHARACTER NER AUDIT COMPLETE')
    print(f'Characters in master: {len(characters)}')
    print(f'Utterances processed: {len(utterances)}')
    print(f'Canonical alias occurrences matched: {len(detail)}')
    print(f'Report: {output_path.resolve()}')
    print('Reminder: this is diagnostic comparison, not an accuracy score.')

if __name__ == '__main__':
    main()
