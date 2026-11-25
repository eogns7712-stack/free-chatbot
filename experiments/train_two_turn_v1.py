"""기존 단문 유지 + 2턴 문답 추가 학습. 후속 답변 손실로 회차 선택."""

# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from two_turn_runtime import TwoTurnModel, encode_record
from train_sns_30min import make_batch, validation_loss


def read(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--data',type=Path,default=Path('data/two_turn_dialogue_v1'))
    p.add_argument('--output',type=Path,default=Path('artifacts/two-turn-dialogue-v1'))
    p.add_argument('--epochs',type=int,default=6)
    p.add_argument('--learning-rate',type=float,default=0.0001)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if a.epochs<1 or not 0<a.learning_rate<float('inf'):p.error('회차와 학습률은 양수여야 합니다.')
    random.seed(42);torch.manual_seed(42)
    engine=TwoTurnModel(a.source);model,tok=engine.model,engine.tok
    records=read(a.data/'train.jsonl')
    valrecords=read(a.data/'validation.jsonl')
    if any(len(r['messages'])!=4 for r in valrecords):raise ValueError('검증은 2턴 대화여야 합니다.')
    train=[encode_record(r,tok) for r in records]
    validation=[encode_record(r,tok,final_only=True) for r in valrecords]
    probes=read(a.data/'probes.jsonl')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
    pad=engine.ids['pad']
    a.output.mkdir(parents=True)
    report={'source':str(a.source),'train_count':len(train),'validation_count':len(validation),
            'learning_rate':a.learning_rate,'seed':42,'batch_size':1,
            'selection':'검증 대화의 마지막 assistant 답변 토큰 손실(EOS 포함)',
            'hashes':{n:hashlib.sha256((a.data/n).read_bytes()).hexdigest() for n in ('train.jsonl','validation.jsonl','probes.jsonl')},
            'best_epoch':0,'best_validation_loss':None,'history':[]}

    def snapshot(epoch):
        rng=torch.get_rng_state();prng=random.getstate()
        try:
            model.eval()
            value=validation_loss(model,validation,pad,float('inf'))
            folder=a.output/'epochs'/f'epoch-{epoch:02d}';folder.mkdir(parents=True)
            model.save_pretrained(str(folder));tok.save(str(folder/'tokenizer.json'))
            answers=[]
            for r in probes:
                messages=r['messages']
                answer=engine.generate(messages[:-1])
                # 같은 마지막 질문만 전달한 결과도 함께 저장하여 문맥 사용 여부를 비교합니다.
                alone=engine.generate([messages[-2]]) if len(messages)==4 else answer
                first=engine.generate(messages[:1]) if len(messages)==4 else None
                rollout=engine.generate([messages[0],{'role':'assistant','content':first},messages[2]]) if len(messages)==4 else answer
                answers.append({**r,'reference':messages[-1]['content'],'answer':answer,
                                'without_history_answer':alone,'generated_first_answer':first,
                                'rollout_answer':rollout,'decision':'pending'})
            (folder/'test_answers.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
            if report['best_validation_loss'] is None or value<report['best_validation_loss']:
                report['best_validation_loss']=value;report['best_epoch']=epoch
            return value
        finally:
            torch.set_rng_state(rng);random.setstate(prng)

    initial=snapshot(0);report['initial_validation_loss']=initial
    (a.output/'training_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'학습 {len(train)}개 | 후속 답변 검증 {len(validation)}개 | 시작 손실 {initial:.4f}',flush=True)
    for epoch in range(1,a.epochs+1):
        model.train();order=list(range(len(train)));random.shuffle(order)
        total=tokens=0
        for i in order:
            ids,labels,mask,count=make_batch(train,[i],pad)
            optimizer.zero_grad();loss=model(input_ids=ids,attention_mask=mask,labels=labels).loss
            if not torch.isfinite(loss):raise RuntimeError('유한하지 않은 손실')
            loss.backward();optimizer.step();total+=loss.item()*count;tokens+=count
        value=snapshot(epoch)
        report['history'].append({'epoch':epoch,'train_loss':total/tokens,'followup_validation_loss':value})
        (a.output/'training_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'{epoch}/{a.epochs}회 | 학습 {total/tokens:.4f} | 후속 답변 검증 {value:.4f}',flush=True)
    print(f"완료 | 선택 회차 {report['best_epoch']} | epochs/epoch-{report['best_epoch']:02d}",flush=True)


if __name__=='__main__':main()
