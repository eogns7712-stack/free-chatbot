"""기존339개 보존 + 제공된 후속 정보 반영 실험. 평가 문장 복사 금지."""
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from three_turn_runtime import encode_record


def read(p):
    return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def key(messages):
    return tuple(''.join(unicodedata.normalize('NFKC', m['content']).split()).casefold()
                 for m in messages if m['role'] == 'user')


def render(form, entity):
    last = ord(entity[-1])
    final = (last-0xAC00)%28 if 0xAC00 <= last <= 0xD7A3 else 0
    text = form.format(x=entity)
    if final:
        text = text.replace(entity+'야.', entity+'이야.').replace(entity+'라고', entity+'이라고')
        text = text.replace(entity+'를', entity+'을').replace(entity+'가 ', entity+'이 ')
        if final != 8:
            text = text.replace(entity+'로', entity+'으로')
    return text


def main():
    source = ROOT/'data/three_turn_flow_v4'
    output = ROOT/'data/followup_information_v5'
    scenario_path = ROOT/'evaluation/followup_information_v5.json'
    if output.exists() or scenario_path.exists():
        raise FileExistsError('기존 실험을 덮어쓰지 않습니다.')
    old = {name: read(source/f'{name}.jsonl') for name in ('train','validation','test','probes')}
    assert len(old['train']) == 339
    tok = Tokenizer.from_file(str(ROOT/'artifacts/three-turn-flow-v4/epochs/epoch-06/tokenizer.json'))
    specs = [
        ('movie', '이번에는 영화 한 편을 감상할 생각이야.', '영화 볼 계획이구나. 어떤 작품이야?',
         ['라따뚜이','코코','인사이드 아웃','월-E','주토피아','토이 스토리'],
         ['{x}야.', '{x}를 골랐어.', '볼 작품은 {x}야.', '{x}를 감상할 거야.'],
         ['이번에 고른 건 {x}라는 작품이야.', '나는 {x}로 정했거든.'],
         ['제목은 {x}라고 해.', '내가 보겠다는 작품은 {x}야.'],
         '그 영화를 볼 계획이구나.'),
        ('plant', '방에서 화초를 기르고 있어.', '잘 자라고 있네. 어떤 식물이야?',
         ['장미','선인장','몬스테라','봉선화','라벤더','고무나무'],
         ['{x}야.', '{x}를 키워.', '기르는 건 {x}야.', '내 화초는 {x}야.'],
         ['지금 기르는 식물 이름은 {x}야.', '우리 집에는 {x}가 자라.'],
         ['이름을 알려 주자면 {x}야.', '내가 돌보는 화초는 {x}라고 해.'],
         '그 식물을 키우는구나!'),
        ('walk', '조금 있다가 산책을 나가려고.', '산책할 계획이구나. 어디로 갈 거야?',
         ['강변','호숫가','뒷산','운동장','수목원','해변'],
         ['{x}야.', '{x}로 갈 거야.', '갈 곳은 {x}야.', '{x} 쪽으로 걸을 거야.'],
         ['산책 장소는 {x}로 정했어.', '나는 {x} 쪽에 가려고 해.'],
         ['목적지는 {x}라고 하면 돼.', '걸으러 가려는 곳은 {x}야.'],
         '그곳으로 산책을 갈 계획이구나.'),
    ]
    new = {s: [] for s in ('train','validation','test')}
    for topic, question, ask, entities, train_forms, val_forms, test_forms, reply in specs:
        for split, names, forms in [('train',entities[:4],train_forms),
                                    ('validation',entities[4:5],val_forms),
                                    ('test',entities[5:],test_forms)]:
            for entity in names:
                for form in forms:
                    new[split].append({'id':f'info-{topic}-{split}-{len(new[split]):03d}',
                        'group':f'information_{topic}', 'loss_scope':'last_assistant',
                        'messages':[{'role':'user','content':question}, {'role':'assistant','content':ask},
                                    {'role':'user','content':render(form, entity)},
                                    {'role':'assistant','content':reply}]})
    # 기존 평가를 학습에 사용하지 않고 사용자 발화 순서 중복을 검사한다.
    frozen = json.loads((ROOT/'evaluation/fresh_dialogues_v1.json').read_text(encoding='utf-8-sig'))
    forbidden = {key([{'role':'user','content':s} for s in r['user_messages']]) for r in frozen}
    all_old = {key(r['messages']) for split in old.values() for r in split}
    seen = set()
    for split, rows in new.items():
        for r in rows:
            k = key(r['messages'])
            assert k not in forbidden | all_old | seen, (split, r['id'])
            seen.add(k)
            ids, labels = encode_record(r, tok, final_only=True)
            prefix_len = len(ids)-len(tok.encode(r['messages'][-1]['content']).ids)-1
            assert (labels[:prefix_len] == -100).all()
            assert labels[-1].item() == tok.token_to_id('<eos>')
            assert len(ids)-len(tok.encode(r['messages'][-1]['content']).ids)-1+48 <= 128
    merged = {s:old[s]+new[s] for s in ('train','validation','test')}
    # 기존 손실 마스킹을 유지하고 전체 레코드를 검사한다.
    for rows in merged.values():
        for r in rows:
            encode_record(r,tok,final_only=r.get('loss_scope')=='last_assistant')
    merged['probes'] = old['probes'] + new['validation'] + new['test']
    scenarios = [{'id':r['id'],'group':r['group'],
        'user_messages':[m['content'] for m in r['messages'] if m['role']=='user'],
        'criteria':'첫 질문은 허용. 제공한 작품명·식물명·장소를 다시 묻지 않고 관련된 답변을 한다.'}
        for r in new['validation']+new['test']]
    output.mkdir()
    for split, rows in merged.items():
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    scenario_path.write_text(json.dumps(scenarios,ensure_ascii=False,indent=2),encoding='utf-8')
    report = {'counts':{s:len(v) for s,v in merged.items()},
              'added':{s:len(v) for s,v in new.items()},
              'source_train_sha256':hashlib.sha256((source/'train.jsonl').read_bytes()).hexdigest(),
              'checks':'기존339개 순서·내용 보존, 신규 사용자 순서 중복 없음, 역할·길이·마스킹 검사 통과',
              'limitations':'정규화 정확 중복 검사이며 의미적 독립성을 보장하지 않음. 제한된 세 주제 실험.'}
    (output/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()

