from pathlib import Path
from tokenizers import Tokenizer
import collections
import hashlib
import json
import re

ROOT = Path('/Users/zhangyunjian/Documents/Codex/2026-11-07/jia')
BASE = ROOT / 'outputs/data_pipeline'
SOURCE = BASE / 'staging/round11_punctuation/train'
TARGET = BASE / 'staging/round11/train'
MAPPING_SOURCE = BASE / 'scripts/topic_mapping_source'
NAMES = ('natural_zh', 'natural_en', 'article_zh', 'article_en')

zh_topics = json.loads((MAPPING_SOURCE / 'topic_zh_input.json').read_text())
en_topics = []
for i in range(10):
    part = (MAPPING_SOURCE / f'topic_en_manual_{i}.txt').read_text().splitlines()
    assert len(part) == (100 if i < 9 else 32)
    en_topics += part
assert len(zh_topics) == len(en_topics) == 932
assert len(set(zh_topics)) == len(zh_topics)
assert len(set(en_topics)) == len(en_topics)
assert all(re.fullmatch(r'[a-z][a-z0-9_]*', x) for x in en_topics)
mapping = dict(zip(zh_topics, en_topics))

TARGET.mkdir(parents=True, exist_ok=True)
changed = collections.Counter()
manifest = {'input_staging': str(SOURCE), 'output_staging': str(TARGET), 'topic_labels_translated': len(mapping), 'files': {}}
for name in NAMES:
    rows = [json.loads(line) for line in (SOURCE / f'{name}.jsonl').open()]
    if name == 'natural_zh':
        for row in rows:
            if row['topic'] in mapping:
                row['topic_zh'] = row['topic']
                row['topic'] = mapping[row['topic']]
                changed[row['topic_zh']] += 1
        assert all(not re.search(r'[\u3400-\u9fff]', row['topic']) for row in rows)
    out = TARGET / f'{name}.jsonl'
    out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    manifest['files'][name] = {'records': len(rows), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
assert sum(changed.values()) == 934
assert set(changed) == set(mapping)

map_path = BASE / 'reports/topic_translation_map_round11.jsonl'
map_path.write_text(''.join(json.dumps({'topic_zh': zh, 'topic_en': mapping[zh], 'records': changed[zh], 'method': 'manual_contextual_translation'}, ensure_ascii=False) + '\n' for zh in zh_topics))
manifest['topic_mapping_sha256'] = hashlib.sha256(map_path.read_bytes()).hexdigest()
manifest['topic_records_translated'] = sum(changed.values())
manifest['total_records'] = sum(x['records'] for x in manifest['files'].values())

tokenizer = Tokenizer.from_file('/Users/zhangyunjian/felixspace/rat/gpt2/tokenizer/tokenizer.json')
tokenizer.no_truncation()
for name in NAMES:
    rows = [json.loads(line) for line in (TARGET / f'{name}.jsonl').open()]
    manifest['files'][name]['tokens'] = sum(len(tokenizer.encode(row['text'], add_special_tokens=False).ids) for row in rows)
manifest['total_tokens'] = sum(x['tokens'] for x in manifest['files'].values())
(BASE / 'reports/staging_round11_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({'topic_labels': len(mapping), 'topic_records': sum(changed.values()), 'records': manifest['total_records'], 'tokens': manifest['total_tokens']}, ensure_ascii=False))
