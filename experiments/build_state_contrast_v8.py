"""산책·영화 상태 대비. 기존 데이터와 평가 입력을 보존한다."""
import hashlib
import json
import sys
import unicodedata
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tokenizers import Tokenizer
from three_turn_runtime import encode_record


def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def norm(s):return ''.join(unicodedata.normalize('NFKC',s).split()).casefold()
def key(ms):return tuple(norm(m['content']) for m in ms if m['role']=='user')


def main():
    base=ROOT/'data/followup_switch_v6';out=ROOT/'data/state_contrast_v8'
    if out.exists():raise FileExistsError('기존 출력 보존')
    old={s:read(base/f'{s}.jsonl') for s in ('train','validation','test','probes')}
    specs={
      'walk':{
       'plan':['내일은 산책을 하러 나갈 생각이야.','조만간 가볍게 산책할 계획을 잡았어.','잠시 뒤 동네 산책을 시작하려고 해.','오늘 오후 산책을 나가기로 정했네.','다음 휴일에 산책을 해 볼 작정이야.'],
       'ongoing':['산책을 시작해서 지금 걷고 있는 중이야.','밖에서 산책하고 있어. 아직 집으로 안 왔어.','지금 산책길을 따라 천천히 걸어가는 중이야.','집 밖에서 한창 산책하고 있네.','산책하는 도중에 잠깐 메시지를 보내.'],
       'done':['산책을 마무리하고 집에 도착했네.','오늘 산책은 끝났어. 지금은 집에서 쉬어.','걷기를 끝내고 방금 산책에서 돌아온 참이야.','산책을 끝내고 귀가해서 앉아 있어.','바깥 산책을 마친 뒤 집에서 쉬는 중이야.'],
       'cancel':['산책하려다 마음을 바꿨어. 오늘은 안 나가.','오늘 잡았던 산책 계획을 취소했네.','산책을 가지 않기로 했어. 집에 있을래.','예정했던 산책을 안 하기로 바꿨어.','산책 계획을 접었어. 오늘은 나가지 않아.']},
      'movie':{
       'plan':['내일 영화 한 편 감상할 생각이야.','곧 영화를 보기 시작할 계획이야.','잠시 뒤 영화 감상을 시작하려고 해.','휴일에 영화 한 편 보기로 정했네.','저녁 식사 후 영화를 볼 작정이야.'],
       'ongoing':['영화를 틀어서 지금 감상하고 있는 중이야.','지금 영화가 상영 중이야. 아직 다 안 봤어.','영화를 보는 도중에 잠깐 메시지를 보내.','한창 영화 감상 중이라 아직 결말은 몰라.','지금 보고 있는 영화가 아직 끝나지 않았네.'],
       'done':['영화 감상을 마무리했어. 끝까지 다 봤네.','영화를 모두 보고 이제 쉬고 있어.','방금 영화의 마지막 장면까지 보고 끝냈어.','영화를 다 감상해서 지금은 끝난 상태야.','마지막까지 영화를 보고 나서 쉬는 중이야.'],
       'cancel':['영화를 보려다 마음을 바꿨어. 오늘은 안 봐.','오늘 잡았던 영화 감상 계획을 취소했네.','영화를 보지 않기로 했어. 다른 일을 할래.','예정했던 영화 감상을 안 하기로 바꿨어.','영화 볼 계획을 접었어. 오늘은 감상하지 않아.']}}
    replies={'walk':{'plan':'산책할 계획이구나.','ongoing':'산책하고 있구나. 천천히 걸어.','done':'산책하고 왔구나. 기분이 어때?','cancel':'산책을 취소했구나. 집에서 편하게 쉬어.'},
             'movie':{'plan':'영화 볼 계획이구나.','ongoing':'영화를 보고 있구나. 재미있게 봐.','done':'재미있게 봤구나. 어떤 장면이 좋았어?','cancel':'영화를 안 보기로 했구나. 편하게 쉬어.'}}
    # 감상 완료만으로 재미를 단정하지 않는 중립 답변.
    replies['movie']['done']='영화를 다 봤구나. 감상이 어땠어?'
    contexts={'walk':[('user','이따 산책하러 나가 볼까 해.'),('assistant','산책할 계획이구나. 어디로 갈 거야?'),('user','동네 둘레길로 정했어.'),('assistant','그곳으로 산책을 갈 계획이구나.')],
              'movie':[('user','이따 영화 감상을 해 볼까 해.'),('assistant','영화 볼 계획이구나. 어떤 작품이야?'),('user','알라딘을 골랐어.'),('assistant','그 영화를 볼 계획이구나.')]}
    new={s:[] for s in ('train','validation','test')}
    frozen=json.loads((ROOT/'evaluation/state_transition_v8.json').read_text(encoding='utf-8'))
    forbidden={norm(q) for c in frozen for q in c['user_messages']}
    seen={key(r['messages']) for rows in old.values() for r in rows}
    for topic,states in specs.items():
      for state,texts in states.items():
       for i,text in enumerate(texts):
        assert norm(text) not in forbidden
        split='train' if i<3 else 'validation' if i==3 else 'test'
        for mode in ('single','history'):
         ms=([{'role':r,'content':s} for r,s in contexts[topic]] if mode=='history' else [])
         ms += [{'role':'user','content':text},{'role':'assistant','content':replies[topic][state]}]
         row={'id':f'state8-{topic}-{state}-{split}-{i}-{mode}','kind':'state_contrast','topic':topic,'state':state,'mode':mode,'messages':ms,'loss_scope':'last_assistant'}
         k=key(ms);assert k not in seen,(split,row['id']);seen.add(k)
         new[split].append(row)
    tok=Tokenizer.from_file(str(ROOT/'artifacts/three-turn-flow-v4/epochs/epoch-06/tokenizer.json'))
    for rows in new.values():
      for r in rows:
       ids,labels=encode_record(r,tok,final_only=True)
       n=len(tok.encode(r['messages'][-1]['content']).ids)+1
       assert (labels[:-n]==-100).all() and labels[-n:].equal(ids[-n:])
       assert len(ids)-n+48<=128
    merged={'train':old['train']+new['train'],
            'validation':old['validation']+[r for r in new['validation'] if r['mode']=='history'],
            'test':old['test']+new['test'],
            'probes':old['probes']+new['validation']+new['test']}
    assert len(merged['train'])==483 and merged['train'][:435]==old['train']
    out.mkdir()
    for split,rows in merged.items():
      (out/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    report={'counts':{s:len(v) for s,v in merged.items()},'added_training':len(new['train']),
            'base_train_sha256':hashlib.sha256((base/'train.jsonl').read_bytes()).hexdigest(),
            'checks':'원본435개 보존. 신규 사용자 순서 중복0개. 기존 상태 평가 문장 제외. 역할·토큰 길이·마스킹 통과.',
            'validation_note':'학습기 제약으로 검증 손실은 기존93+상태 문맥8개. 단문8개는probes에 포함.',
            'scope':'두 주제·네 상태. 완료/취소는 계획에서 바뀐 문맥도 포함. 상태 전이 전체 조합 아님.'}
    (out/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
