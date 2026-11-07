from pathlib import Path
import hashlib
import json
import re

BASE = Path('/Users/zhangyunjian/Documents/Codex/2026-11-07/jia/outputs/data_pipeline')
SOURCE = BASE / 'staging/round10/train'
TARGET = BASE / 'staging/round11_punctuation/train'
LOG = BASE / 'reports/punctuation_decisions_round11.jsonl'
NAMES = ('natural_zh', 'natural_en', 'article_zh', 'article_en')
ZH_PUNCT_SPACE = re.compile(r'([，。！？；：、）》】」』”’])[ \t]+')

TARGET.mkdir(parents=True, exist_ok=True)
decisions = []
counts = {}
for name in NAMES:
    rows = [json.loads(line) for line in (SOURCE / f'{name}.jsonl').open()]
    changed = 0
    removed = 0
    for row in rows:
        if name.endswith('_zh'):
            before = row['text']
            after, n = ZH_PUNCT_SPACE.subn(r'\1', before)
            if n:
                row['text'] = after
                changed += 1
                removed += n
                decisions.append({
                    'record_id': row['id'],
                    'rule': 'remove_horizontal_space_after_chinese_punctuation',
                    'replacements': n,
                    'old_text_sha256': hashlib.sha256(before.encode()).hexdigest(),
                    'new_text_sha256': hashlib.sha256(after.encode()).hexdigest(),
                    'old_text': before,
                    'new_text': after,
                })
    path = TARGET / f'{name}.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    counts[name] = {'records': len(rows), 'changed_records': changed, 'removed_spaces': removed}
LOG.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in decisions))
print(json.dumps(counts, ensure_ascii=False))
