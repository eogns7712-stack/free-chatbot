"""신규48개만 마지막 답변 학습에서 전체 assistant 학습으로 변경한다."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from three_turn_runtime import encode_record


def main():
    source = ROOT/'data/followup_information_v5'
    target = ROOT/'data/followup_information_v5_all_assistant'
    if target.exists():
        raise FileExistsError(f'기존 출력 보존: {target}')
    rows = [json.loads(s) for s in (source/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 387
    tok = Tokenizer.from_file(str(ROOT/'artifacts/three-turn-flow-v4/epochs/epoch-06/tokenizer.json'))
    output, changes = [], []
    for i, row in enumerate(rows):
        new = dict(row)
        if str(row.get('id','')).startswith('info-'):
            assert i >= 339 and row.get('loss_scope') == 'last_assistant'
            new.pop('loss_scope')
            old_ids, old_labels = encode_record(row,tok,final_only=True)
            new_ids, new_labels = encode_record(new,tok,final_only=False)
            assert old_ids.equal(new_ids)
            # 마지막 답변의 마스킹은 동일하다. 첫 assistant 본문과EOS만 추가한다.
            first = row['messages'][1]['content']
            expected = len(tok.encode(first).ids)+1
            assert int((new_labels != -100).sum()-(old_labels != -100).sum()) == expected
            for label in (old_labels, new_labels):
                assert label[-1].item() == tok.token_to_id('<eos>')
            changes.append({'index':i,'id':row['id'],'added_supervised_tokens':expected})
        else:
            assert row == new
        output.append(new)
    assert len(changes) == 48 and output[:339] == rows[:339]
    target.mkdir()
    (target/'train.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in output),encoding='utf-8')
    hashes = {}
    for name in ('validation.jsonl','test.jsonl','probes.jsonl'):
        content = (source/name).read_bytes()
        (target/name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    report = {'source':str(source),'changed_records':48,'unchanged_original_records':339,
        'only_change':'신규48개에서 loss_scope 제거. 첫 assistant 본문과EOS도 학습.',
        'train_order_and_messages_identical':True,'evaluation_files_byte_identical':hashes,
        'source_train_sha256':hashlib.sha256((source/'train.jsonl').read_bytes()).hexdigest(),
        'changes':changes}
    (target/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'완료 | 기존339개 보존 | 신규48개 학습 범위 변경 | {target}')


if __name__ == '__main__':
    main()

