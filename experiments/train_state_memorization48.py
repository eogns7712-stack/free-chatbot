"""상태 학습48개 습득 진단. 기존 능력 유지나 일반화 평가가 아니다."""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(r'C:\Users\510\Desktop\free-chatbot'))
    p.add_argument('--source',type=Path,default=Path('artifacts/state-contrast-v8/epochs/epoch-02'))
    p.add_argument('--output',type=Path,default=Path('artifacts/state-memorization48-v1'))
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--minutes',type=float,default=20)
    p.add_argument('--learning-rate',type=float,default=0.0001)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    if a.epochs<1 or not 0<a.minutes<float('inf') or not 0<a.learning_rate<float('inf'):p.error('회차·시간·학습률은 양수여야 합니다.')
    source=a.source if a.source.is_absolute() else a.root/a.source
    output=a.output if a.output.is_absolute() else a.root/a.output
    if not a.check_only and output.exists():raise FileExistsError(f'기존 결과 보존: {output}')
    sys.path.insert(0,str(a.root))
    import torch
    from three_turn_runtime import ThreeTurnModel,encode_record
    random.seed(42);torch.manual_seed(42)
    engine=ThreeTurnModel(source);model,tok=engine.model,engine.tok
    data=a.root/'data/state_contrast_v8/train.jsonl'
    rows=[json.loads(s) for s in data.read_text(encoding='utf-8').splitlines() if s.strip()]
    rows=[r for r in rows if str(r.get('id','')).startswith('state8-')]
    assert len(rows)==48
    examples=[]
    for r in rows:
        x,y=encode_record(r,tok,final_only=True)
        n=len(tok.encode(r['messages'][-1]['content']).ids)+1
        assert (y[:-n]==-100).all() and y[-n:].equal(x[-n:])
        assert len(x)-n+48<=128
        examples.append((x,y))
    if a.check_only:
        model.train()
        x,y=examples[-1]
        loss=model(input_ids=x[None],attention_mask=torch.ones_like(x[None]),labels=y[None]).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert any(v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0 for v in model.parameters())
        print(f'48개 역할·길이·마스킹 및 역전파 검사 통과 | 손실 {loss.item():.4f} | 저장·학습 스텝 없음')
        return
    output.mkdir(parents=True)
    (output/'train48.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate)
    start=time.perf_counter();deadline=start+a.minutes*60
    report={'source':str(source),'data_sha256':hashlib.sha256(data.read_bytes()).hexdigest(),
        'source_weights':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.safetensors')},
        'seed':42,'threads':4,'batch_size':1,'learning_rate':a.learning_rate,
        'max_epochs':a.epochs,'minutes':a.minutes,'count':48,'history':[],
        'limitations':'학습 예시 암기 진단. 선택 모델을 채팅 기준으로 교체하지 않음. 기존 기능 퇴보 가능. resume 없음.'}
    best=None
    def evaluate(epoch):
        nonlocal best
        rng=torch.get_rng_state();prng=random.getstate();model.eval()
        answers=[];total=0;tokens=0
        with torch.inference_mode():
            for r,(x,y) in zip(rows,examples):
                count=int((y[1:]!=-100).sum())
                loss=model(input_ids=x[None],attention_mask=torch.ones_like(x[None]),labels=y[None]).loss
                assert torch.isfinite(loss)
                total+=loss.item()*count;tokens+=count
                prompt=engine.prompt_ids(r['messages'][:-1]);t=torch.tensor([prompt])
                out=model.generate(input_ids=t,attention_mask=torch.ones_like(t),max_new_tokens=48,
                    do_sample=False,num_beams=1,repetition_penalty=1.0,eos_token_id=engine.ids['eos'],pad_token_id=engine.ids['pad'])
                actual=out[0,len(prompt):].tolist()
                expected=tok.encode(r['messages'][-1]['content']).ids+[engine.ids['eos']]
                history=[];turns=[]
                for i in range(0,len(r['messages']),2):
                    q=r['messages'][i];ans=engine.generate(history[-4:]+[q])
                    history.extend([q,{'role':'assistant','content':ans}]);turns.append({'user':q['content'],'assistant':ans})
                reference=r['messages'][-1]['content']
                answers.append({'id':r['id'],'reference':reference,
                    'gold_answer':tok.decode(actual,skip_special_tokens=True).strip(),
                    'token_match_including_eos':actual==expected,
                    'rollout_last_match':turns[-1]['assistant']==reference,'rollout_turns':turns})
        summary={'epoch':epoch,'fit_loss':total/tokens,
            'token_match_including_eos':sum(x['token_match_including_eos'] for x in answers),
            'rollout_last_match':sum(x['rollout_last_match'] for x in answers),
            'elapsed_minutes':(time.perf_counter()-start)/60}
        report['history'].append(summary)
        (output/f'answers-epoch-{epoch:03d}.json').write_text(json.dumps(answers,ensure_ascii=False,indent=2),encoding='utf-8')
        score=(summary['token_match_including_eos'],-summary['fit_loss'])
        if best is None or score>best:
            best=score;folder=output/'best';model.save_pretrained(str(folder));tok.save(str(folder/'tokenizer.json'));report['best_epoch']=epoch
        (output/'training_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        torch.set_rng_state(rng);random.setstate(prng)
        print(json.dumps(summary,ensure_ascii=False),flush=True)
        return summary['token_match_including_eos']==48 and summary['rollout_last_match']==48
    success=evaluate(0);stop='fit_complete' if success else 'max_epochs'
    for epoch in range(1,a.epochs+1):
        if success:break
        model.train();order=list(range(48));random.shuffle(order);complete=True
        for i in order:
            if time.perf_counter()>=deadline:complete=False;break
            x,y=examples[i];optimizer.zero_grad()
            loss=model(input_ids=x[None],attention_mask=torch.ones_like(x[None]),labels=y[None]).loss
            if not torch.isfinite(loss):raise RuntimeError('유한하지 않은 손실')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        if epoch%5==0 or epoch==a.epochs or not complete:
            success=evaluate(epoch)
            if success:stop='fit_complete';break
        if not complete:stop='time_budget';break
    report['stop_reason']=stop
    assert report['source_weights']=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in source.glob('*.safetensors')}
    (output/'training_log.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'완료 | {stop} | {output}',flush=True)


if __name__=='__main__':main()
