"""기존 문답 유지 + 전환 후 후속 응답까지 최대3턴 학습."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from three_turn_runtime import ThreeTurnModel, encode_record
from train_sns_30min import make_batch, validation_loss


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only',action='store_true',help='마스킹만 검사하고 학습·저장하지 않음')
    p.add_argument('--first-reply-cap',type=int,choices=(0,1),default=1,help='0=원본 학습, 1=동일 첫 문답의 첫 응답을 한 레코드에서만 학습')
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/followup_switch_v6'))
    p.add_argument('--output',type=Path,default=Path('artifacts/first-reply-cap-v7'))
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--learning-rate',type=float,default=0.0001)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.epochs<1 or not 0<a.learning_rate<float('inf'):p.error('회차와 학습률은 양수여야 합니다.')
    random.seed(42);torch.manual_seed(42)
    engine=ThreeTurnModel(a.source);model,tok=engine.model,engine.tok
    records=read(a.data/'train.jsonl');valrecords=read(a.data/'validation.jsonl');probes=read(a.data/'probes.jsonl')
    if any(len(r['messages']) not in (4,6) for r in valrecords):raise ValueError('검증은2턴 또는3턴 대화여야 합니다.')
    train=[encode_record(r,tok,final_only=r.get('loss_scope')=='last_assistant') for r in records]
    # 동일 첫 사용자+assistant 토큰열별 대표 레코드 선택. 단문을 우선 보존한다.
    groups={}
    for i,r in enumerate(records):
        if r.get('loss_scope')=='last_assistant' and len(r['messages'])>2:
            continue
        first=r['messages'][:2]
        key=tuple((m['role'],tuple(tok.encode(m['content']).ids)) for m in first)
        groups.setdefault(key,[]).append(i)
    modified=[];before=sum(int((y!=-100).sum()) for _,y in train)
    if a.first_reply_cap:
        for indices in groups.values():
            keep=next((i for i in indices if len(records[i]['messages'])==2),indices[0])
            for i in indices:
                if i==keep:continue
                assert len(records[i]['messages'])>2, '중복 단문은 별도 처리 필요'
                x,y=train[i];y=y.clone()
                first_length=sum(len(tok.encode(m['content']).ids)+2 for m in records[i]['messages'][:2])
                y[:first_length]=-100
                assert (y!=-100).any(), '학습 목표가 없는 레코드'
                # 첫 문답 이후 목표는 입력·마스킹 모두 그대로 보존한다.
                assert y[first_length:].equal(train[i][1][first_length:])
                train[i]=(x,y);modified.append(i)
    after=sum(int((y!=-100).sum()) for _,y in train)
    masking={'first_reply_cap':a.first_reply_cap,'modified_records':modified,
             'supervised_tokens_before':before,'supervised_tokens_after':after,
             'first_reply_groups':len(groups),'policy':'동일 토큰 첫 문답마다 한 레코드/매회. 기존 last_assistant 보존'}
    if a.check_only:
        print(json.dumps(masking,ensure_ascii=False,indent=2));return
    validation=[encode_record(r,tok,final_only=True) for r in valrecords]
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate);pad=engine.ids['pad']
    a.output.mkdir(parents=True)
    report={'masking':masking,'source':str(a.source),'train_count':len(train),'validation_count':len(validation),
            'learning_rate':a.learning_rate,'seed':42,'batch_size':1,'max_history_pairs':2,
            'selection':'검증 마지막 assistant 답변의 토큰 손실. 정답 이전 응답을 문맥으로 제공.',
            'hashes':{n:hashlib.sha256((a.data/n).read_bytes()).hexdigest() for n in ('train.jsonl','validation.jsonl','probes.jsonl')},
            'best_epoch':0,'best_validation_loss':None,'history':[]}

    def snapshot(epoch):
        rng=torch.get_rng_state();prng=random.getstate()
        try:
            model.eval();value=validation_loss(model,validation,pad,float('inf'))
            folder=a.output/'epochs'/f'epoch-{epoch:02d}';folder.mkdir(parents=True)
            model.save_pretrained(str(folder));tok.save(str(folder/'tokenizer.json'))
            answers=[]
            for r in probes:
                messages=r['messages']
                answer=engine.generate(messages[:-1])
                alone=engine.generate([messages[-2]]) if len(messages)>2 else answer
                generated=[];turns=[]
                for i in range(0,len(messages),2):
                    q=messages[i]
                    out=engine.generate([*generated[-4:],q])
                    generated.extend([q,{'role':'assistant','content':out}])
                    turns.append({'user':q['content'],'assistant':out,'reference':messages[i+1]['content']})
                answers.append({**r,'reference':messages[-1]['content'],'answer':answer,
                                'without_history_answer':alone,'rollout_answer':turns[-1]['assistant'],
                                'rollout_turns':turns,'decision':'pending'})
            (folder/'test_answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
            if report['best_validation_loss'] is None or value<report['best_validation_loss']:
                report['best_validation_loss']=value;report['best_epoch']=epoch
            return value
        finally:
            torch.set_rng_state(rng);random.setstate(prng)

    def save_log():
        (a.output/'training_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    initial=snapshot(0);report['initial_validation_loss']=initial;save_log()
    print(f'학습 {len(train)}개 | 검증 {len(validation)}개 | 시작 {initial:.4f}',flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();order=list(range(len(train)));random.shuffle(order)
        total=tokens=0
        for i in order:
            ids,labels,mask,count=make_batch(train,[i],pad)
            optimizer.zero_grad();loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
            if not torch.isfinite(loss):raise RuntimeError('유한하지 않은 손실')
            loss.backward();optimizer.step();total+=loss.item()*count;tokens+=count
        value=snapshot(epoch)
        report['history'].append({'epoch':epoch,'train_loss':total/tokens,'followup_validation_loss':value});save_log()
        print(f'{epoch}/{a.epochs}회 | 학습 {total/tokens:.4f} | 후속 검증 {value:.4f}',flush=True)
    print(f"완료 | 선택 {report['best_epoch']}회차",flush=True)


if __name__=='__main__':main()
