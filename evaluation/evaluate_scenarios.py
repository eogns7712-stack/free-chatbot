"""고정된 시나리오를 현재3턴 앱과 같은 방식으로 평가. 추가 학습 없음."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from three_turn_runtime import ThreeTurnModel


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=PROJECT_ROOT/'artifacts/three-turn-flow-v4/epochs/epoch-06')
    parser.add_argument('--scenarios', type=Path, default=Path(__file__).with_name('fresh_dialogues_v1.json'))
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'reports/fresh_dialogues_v1')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'기존 결과 보존. --output으로 새 폴더를 지정하세요: {args.output}')
    scenarios = json.loads(args.scenarios.read_text(encoding='utf-8-sig'))
    if not scenarios or len({r['id'] for r in scenarios}) != len(scenarios):
        raise ValueError('시나리오가 없거나 ID가 중복됩니다.')
    if any(not 1 <= len(r['user_messages']) <= 3 for r in scenarios):
        raise ValueError('이 평가는 최대3턴 대화용입니다.')
    weights = sorted(args.model.glob('*.safetensors'))
    if not weights:
        raise FileNotFoundError('로컬 모델 가중치가 없습니다.')
    before = {p.name:file_hash(p) for p in weights}
    args.output.mkdir(parents=True)
    # 생성 전 평가 입력 고정. 모든 모델은 같은 파일을 사용합니다.
    (args.output/'scenarios.json').write_bytes(args.scenarios.read_bytes())
    engine = ThreeTurnModel(args.model)
    results = []
    for case in scenarios:
        history, turns = [], []
        for i, question in enumerate(case['user_messages'],1):
            context = history[-4:] + [{'role':'user','content':question}]
            try:
                answer = engine.generate(context)
            except ValueError as error:
                turns.append({'turn':i,'user':question,'assistant':None,'error':str(error),'input_messages':context})
                break
            turns.append({'turn':i,'user':question,'assistant':answer,'input_messages':context})
            history.extend([{'role':'user','content':question},{'role':'assistant','content':answer}])
        result = {**case,'turns':turns,'decision':'pending','notes':''}
        results.append(result)
        print(json.dumps({'id':case['id'],'group':case['group'],
                          'turns':[{k:t.get(k) for k in ('user','assistant','error') if k in t} for t in turns]},
                         ensure_ascii=False),flush=True)
    assert before == {p.name:file_hash(p) for p in weights}, '평가 중 모델 가중치가 바뀌었습니다.'
    import torch, transformers
    report = {'model':str(args.model.resolve()),'weight_hashes':before,
              'tokenizer_sha256':file_hash(args.model/'tokenizer.json'),
              'scenarios_sha256':file_hash(args.scenarios),'max_history_pairs':2,
              'generation':{'do_sample':False,'num_beams':1,'max_new_tokens':48},
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__},
              'method':'각 시나리오는 새 대화. 모델이 생성한 최근 두 문답만 전달하며 정답 중간 응답을 제공하지 않음.',
              'limitations':'학습 대화와 전체 사용자 발화 순서의 정확 중복을 배제한 소규모 개발 평가. 의미가 유사한 예시와 범위 밖 진단을 포함함.',
              'results':results}
    (args.output/'answers.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'저장: {args.output.resolve()}',flush=True)


if __name__ == '__main__':
    main()
