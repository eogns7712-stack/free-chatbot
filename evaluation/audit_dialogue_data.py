"""문맥 전체 충돌과 문맥에 따른 정상 답변 차이를 구분하는 읽기 전용 검사."""
import argparse
import collections
import hashlib
import json
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def norm(s):
    return ''.join(unicodedata.normalize('NFKC',s).split()).casefold()


def prefix(messages):
    return tuple((m['role'],norm(m['content'])) for m in messages)


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/followup_switch_v6')
    p.add_argument('--output',type=Path,default=ROOT/'reports/followup-switch-v6-data-audit')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError('기존 검사 결과를 보존하세요. 새 --output을 지정하세요.')
    splits={s:read(args.data/f'{s}.jsonl') for s in ('train','validation','test')}
    train=splits['train'];contexts=collections.defaultdict(list);users=collections.defaultdict(list)
    exact=collections.defaultdict(list);last_scope=0;masked_answers=0;supervised=0
    answers=collections.Counter();first_questions=collections.Counter();first_answers=collections.defaultdict(collections.Counter)
    for i,row in enumerate(train):
        ms=row['messages'];exact[prefix(ms)].append(i)
        assert len(ms) in (2,4,6) and [m['role'] for m in ms]==['user','assistant']*(len(ms)//2)
        last_scope+=row.get('loss_scope')=='last_assistant'
        for j,m in enumerate(ms):
            if m['role']!='assistant':continue
            scored=row.get('loss_scope')!='last_assistant' or j==len(ms)-1
            entry={'record':i,'id':row.get('id'),'assistant_index':j,'answer':m['content'],'supervised':scored}
            contexts[prefix(ms[:j])].append(entry)
            users[norm(ms[j-1]['content'])].append(entry)
            if scored:
                supervised+=1;answers[m['content']]+=1
                if j==1:
                    first_questions[ms[0]['content']]+=1
                    first_answers[ms[0]['content']][m['content']]+=1
            else:masked_answers+=1
    def conflict(v):return len({norm(x['answer']) for x in v if x['supervised']})>1
    collisions=[{'context':list(k),'occurrences':v} for k,v in contexts.items() if conflict(v)]
    ambiguous=[{'user_normalized':k,'answers':dict(collections.Counter(x['answer'] for x in v if x['supervised'])),
                'records':sorted({x['record'] for x in v if x['supervised']})}
               for k,v in users.items() if conflict(v)]
    overlaps={}
    for left,right in [('train','validation'),('train','test'),('validation','test')]:
        lk={prefix(x['messages']) for x in splits[left]};rk={prefix(x['messages']) for x in splits[right]}
        lu={tuple(norm(m['content']) for m in x['messages'] if m['role']=='user') for x in splits[left]}
        ru={tuple(norm(m['content']) for m in x['messages'] if m['role']=='user') for x in splits[right]}
        overlaps[f'{left}/{right}']={'whole_dialogue':len(lk&rk),'whole_user_sequence':len(lu&ru)}
    shared={}
    train_prefixes={k for k,v in contexts.items() if any(x['supervised'] for x in v)}
    for split in ('validation','test'):
        ms=[x['messages'] for x in splits[split]]
        shared[split]={'first_user_seen_in_train':sum(prefix(x[:1]) in train_prefixes for x in ms),
            'final_answer_prefix_seen_in_train':sum(prefix(x[:-1]) in train_prefixes for x in ms), 'count':len(ms)}
    endings=[{'record':i,'messages':row['messages']} for i,row in enumerate(train)
             if any(any(w in m['content'] for w in ('다음에 또 봐','이야기 즐거웠어')) for m in row['messages'] if m['role']=='assistant')]
    report={'counts':{s:len(v) for s,v in splits.items()},'last_assistant_records':last_scope,
        'supervised_assistant_occurrences':supervised,'masked_assistant_occurrences':masked_answers,
        'duplicate_dialogue_groups':[{'records':v,'extra_copies':len(v)-1} for v in exact.values() if len(v)>1],
        'conflicting_full_contexts':collisions,
        'same_user_different_answers':ambiguous,
        'supervised_answer_frequency':answers.most_common(),
        'supervised_first_user_frequency':first_questions.most_common(),
        'first_user_answer_maps':{k:dict(v) for k,v in first_answers.items()},
        'split_overlap':overlaps,'shared_prefixes':shared,'ending_records':endings,
        'hashes':{s:hashlib.sha256((args.data/f'{s}.jsonl').read_bytes()).hexdigest() for s in splits},
        'limitations':'공백 제거·NFKC 정확 검사. 의미적 충돌은 자동 확정하지 않음. 같은 사용자 문장의 답변 차이는 문맥 의존일 수 있음. 빈도는 예시 노출 횟수이며 실제 토큰 손실 가중치가 아님.'}
    args.output.mkdir(parents=True)
    (args.output/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('counts','last_assistant_records','supervised_assistant_occurrences','masked_assistant_occurrences','split_overlap','shared_prefixes')},ensure_ascii=False,indent=2))
    print('중복 그룹',len(report['duplicate_dialogue_groups']),'전체 문맥 충돌',len(collisions),'사용자 문장별 답변 차이',len(ambiguous))
    print('상위 답변',answers.most_common(8));print('종료 응답 레코드',len(endings))
    print('상위 첫 사용자',first_questions.most_common(6))


if __name__=='__main__':main()
