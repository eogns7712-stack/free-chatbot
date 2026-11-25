"""같은 직전 문맥에서 후속 정보와 새 주제를 대비하는 v6 데이터."""
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tokenizers import Tokenizer
from three_turn_runtime import encode_record


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def key(messages):
    return tuple(''.join(unicodedata.normalize('NFKC',m['content']).split()).casefold()
                 for m in messages if m['role']=='user')


def main():
    base=ROOT/'data/followup_information_v5_all_assistant'
    output=ROOT/'data/followup_switch_v6'
    scenario=ROOT/'evaluation/followup_switch_v6.json'
    if output.exists() or scenario.exists():
        raise FileExistsError('기존 데이터와 평가 입력을 덮어쓰지 않습니다.')
    old={s:read(base/f'{s}.jsonl') for s in ('train','validation','test','probes')}
    assert len(old['train'])==387
    tok=Tokenizer.from_file(str(ROOT/'artifacts/three-turn-flow-v4/epochs/epoch-06/tokenizer.json'))
    contexts={
        'movie':('이번에는 영화 한 편을 감상할 생각이야.','영화 볼 계획이구나. 어떤 작품이야?'),
        'plant':('방에서 화초를 기르고 있어.','잘 자라고 있네. 어떤 식물이야?'),
        'walk':('조금 있다가 산책을 나가려고.','산책할 계획이구나. 어디로 갈 거야?'),
    }
    targets={
        'movie':(['이따 영화 한 편을 틀어서 보려고.','휴일에는 영화 감상을 할 생각이야.',
                  '오늘은 극장에 가서 한 편 감상할 예정이야.','저녁에는 영화 한 편 감상할 계획을 세웠어.'],
                 '영화 볼 계획이구나. 어떤 작품이야?'),
        'plant':(['새 화분을 사서 식물을 기르기 시작했어.','화초 하나를 사서 집에 들였어.',
                  '작은 화분 하나를 들여서 키우기 시작했네.','새 화초를 장만해 창가에서 키우는 중이야.'],
                 '새 식물을 들였구나. 어떤 식물이야?'),
        'walk':(['이따 동네 길을 산책하러 나가려고.','휴일에는 산책을 나갈 생각이야.',
                 '이번 오후에는 천천히 걸으러 나갈 예정이야.','저녁에는 산책을 하러 밖에 나갈 계획이야.'],
                '산책할 계획이구나. 어디로 갈 거야?'),
        'tired':(['어젯밤에 잠이 모자라서 계속 졸려.','수면 시간이 짧아서 눈이 자꾸 감겨.',
                  '잠을 설쳐서 지금 졸음이 쏟아져.','밤새 잠을 못 이뤄서 지금 눈꺼풀이 무거워.'],
                 '잠이 부족하구나. 잠깐 눈을 붙여.'),
        'greeting':(['다른 주제로 편하게 얘기를 시작하고 싶어.','새로운 이야기를 꺼내고 싶은데 들어 줄래?',
                     '이번에는 다른 이야기를 나눠도 괜찮겠어?','새 얘기를 하고 싶어. 내 말을 들어 줄 수 있어?'],
                    '좋아. 편하게 이야기하자.'),
    }
    entities={
        'movie':[['알라딘','겨울왕국','미니언즈','슈렉'],['센과 치히로','쿵푸팬더','루카','소울'],
                 ['인어공주','엘리멘탈','리오','메리다']],
        'plant':[['아이비','페페로미아','테이블야자','로즈마리'],['스파티필룸','산세베리아','율마','호야'],
                 ['베고니아','동백','해피트리','카라']],
        'walk':[['하천변','학교 앞','마을길','정원'],['산책로','꽃길','언덕길','공원 입구'],
                ['물가','동네 뒷길','나무 그늘길','호수 옆']],
    }
    replies={'movie':'그 영화를 볼 계획이구나.','plant':'그 식물을 키우는구나!',
             'walk':'그곳으로 산책을 갈 계획이구나.'}
    forms={
        'movie':['내가 고른 작품 제목: {x}.','감상하려는 영화 제목은 바로 {x}.',
                 '이번에 선택한 작품의 제목을 말하면 {x}.','볼 영화 제목을 알려 줄게: {x}.'],
        'plant':['내가 기르는 식물 이름: {x}.','돌보고 있는 화초 이름은 바로 {x}.',
                 '우리 집 식물의 이름을 말하면 {x}.','키우는 식물 이름을 알려 줄게: {x}.'],
        'walk':['내가 정한 산책 목적지: {x}.','걸으러 갈 장소는 바로 {x}.',
                '이번 산책의 목적지를 말하면 {x}.','산책 장소를 알려 줄게: {x}.'],
    }
    new={s:[] for s in ('train','validation','test')}
    for source,(q,a) in contexts.items():
        others=[t for t in targets if t!=source]
        for j,target in enumerate(others):
            for split,indices,entity_group in [('train',[0,1],0),('validation',[2],1),('test',[3],2)]:
                for i in indices:
                    pair=f'{source}-{target}-{split}-{i}'
                    for branch in ('followup','switch'):
                        user=(forms[source][i].format(x=entities[source][entity_group][j])
                              if branch=='followup' else targets[target][0][i])
                        reply=replies[source] if branch=='followup' else targets[target][1]
                        new[split].append({'id':f'contrast-{pair}-{branch}',
                            'kind':'contrast','branch':branch,'source_topic':source,
                            'target_topic':source if branch=='followup' else target,'pair_id':pair,
                            'messages':[{'role':'user','content':q},{'role':'assistant','content':a},
                                        {'role':'user','content':user},{'role':'assistant','content':reply}]})
    seen={key(r['messages']) for rows in old.values() for r in rows}
    frozen=json.loads((ROOT/'evaluation/fresh_dialogues_v1.json').read_text(encoding='utf-8-sig'))
    seen.update(key([{'role':'user','content':s} for s in r['user_messages']]) for r in frozen)
    for split,rows in new.items():
        for r in rows:
            k=key(r['messages'])
            assert k not in seen,(split,r['id'])
            seen.add(k)
            ids,labels=encode_record(r,tok)
            # 각 역할 토큰 및 사용자 본문은 마스킹, 두 assistant 본문과EOS는 학습.
            offset=0
            for m in r['messages']:
                count=len(tok.encode(m['content']).ids)+2
                assert labels[offset].item()==-100
                if m['role']=='user':assert (labels[offset:offset+count]==-100).all()
                else:assert labels[offset+1:offset+count].equal(ids[offset+1:offset+count])
                offset+=count
            assert len(ids)-len(tok.encode(r['messages'][-1]['content']).ids)-1+48<=128
    merged={s:old[s]+new[s] for s in ('train','validation','test')}
    merged['probes']=old['probes']+new['validation']+new['test']
    assert merged['train'][:387]==old['train']
    for rows in merged.values():
        for r in rows:encode_record(r,tok,final_only=r.get('loss_scope')=='last_assistant')
    output.mkdir()
    for split,rows in merged.items():
        (output/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    cases=[{'id':r['id'],'group':r['branch'],'pair_id':r['pair_id'],
            'user_messages':[m['content'] for m in r['messages'] if m['role']=='user'],
            'criteria':'후속 정보에는 재질문하지 않고 관련 반응. 전환에는 새 주제에 반응. 첫 응답도 함께 판정.'}
           for r in new['validation']+new['test']]
    scenario.write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'counts':{s:len(rows) for s,rows in merged.items()},'added':{s:len(rows) for s,rows in new.items()},
            'base_train_sha256':hashlib.sha256((base/'train.jsonl').read_bytes()).hexdigest(),
            'checks':'기존387개 보존, 신규 사용자 순서 중복 없음, 역할·길이·손실 마스킹 검사 통과',
            'limitations':'공통 첫 문맥을 공유하는 제한된 대비 실험. 의미적 독립성과 자유 대화 성능을 보장하지 않음.'}
    (output/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
