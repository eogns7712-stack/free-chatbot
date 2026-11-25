"""주제별 고정 단문 응답을 이용한 통제 실험 데이터 생성."""

# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import json
import shutil
from pathlib import Path

ROOT = PROJECT_ROOT
DEST = ROOT / 'short_control_package' / 'data' / 'short_dialogue_control_v1'

# 각 주제: 학습 12, 검증 4, 평가 4. 고정 답변은 의도적으로 사용합니다.
TOPICS = {
 'greeting': ('반가워! 편하게 이야기해.', [
 '안녕, 이야기하러 왔어.', '반가워, 나랑 대화하자.', '안녕하세요, 처음 왔어요.',
 '하이, 잠깐 얘기하자.', '나 왔어, 인사할게.', '처음 뵙겠습니다. 잘 부탁해요.',
 '오랜만이야, 다시 만나서 반가워.', '여기서 대화하고 싶어.', '잠시 말동무가 되어 줄래?',
 '다시 접속했어. 반가워.', '대화를 시작하고 싶어요.', '너랑 수다를 떨고 싶어.',
 '반갑습니다, 잠시 대화할까요?', '인사하려고 찾아왔어요.', '오늘 처음으로 말을 걸어 봐.', '다시 만나니 반갑네.',
 '안녕하신가요? 이야기를 나누고 싶습니다.', '오랜만에 채팅하러 들어왔네.', '말할 사람이 필요해서 찾아왔어.', '처음 오는 채팅방이네. 반갑다.']),
 'tired': ('많이 피곤하구나. 잠깐 쉬어.', [
 '일을 오래 해서 피곤해.', '잠을 적게 자서 졸려.', '하루 종일 움직여서 지쳤어.',
 '공부하다가 기운이 빠졌어.', '야근 때문에 너무 힘들어.', '오늘은 몸이 무겁고 피곤해.',
 '할 일이 많아서 녹초가 됐어.', '청소를 오래 했더니 힘이 없어.', '밤을 새웠더니 졸음이 와.',
 '오래 서 있었더니 다리가 피곤해.', '쉴 틈 없이 일해서 지쳤어.', '잠이 모자라서 눈꺼풀이 무거워.',
 '아침부터 일했더니 기력이 없네.', '어젯밤 거의 못 자서 몹시 졸리다.', '공부를 마치니 진이 빠져.', '오래 움직였더니 쉬고 싶어.',
 '업무가 끝났는데 온몸이 녹초야.', '밤새 깨어 있었더니 눈이 자꾸 감기네.', '집안일을 한참 했더니 체력이 바닥났어.', '오늘 너무 바빠서 이제는 기운이 하나도 없어.']),
 'walk': ('산책하고 왔구나. 기분이 어때?', [
 '공원에서 산책하고 왔어.', '집 근처를 걸어서 한 바퀴 돌았어.', '강가를 따라 걷고 왔어.',
 '저녁에 동네 산책을 했어.', '숲길을 천천히 걷고 왔어.', '점심 먹고 밖에서 좀 걸었어.',
 '아침에 산책로를 다녀왔어.', '친구랑 공원을 걸었어.', '혼자 동네를 걷다 왔어.',
 '호수 주변을 산책했어.', '골목길을 따라서 걷고 왔어.', '바깥 공기를 쐬며 걸었어.',
 '강변 산책로를 다녀왔어요.', '나무가 있는 길을 걸어 봤어.', '집 주변에서 가볍게 산책했네.', '저녁 바람을 맞으며 한 바퀴 걸었어.',
 '공원 길을 천천히 걸어서 돌아왔지.', '오늘은 호숫가 둘레를 걸어 봤네.', '동네 골목을 구경하며 산책하고 귀가했어.', '아침 공기를 마시며 근처 길을 걷고 돌아왔어요.']),
 'movie': ('영화 이야기구나. 어떤 장르가 좋아?', [
 '영화를 보고 싶어.', '코미디 영화를 볼까 해.', '극장에서 영화 한 편 보려고.',
 '주말에 액션 영화를 볼 예정이야.', '집에서 볼 영화를 찾고 있어.', '영화 고르는 중이야.',
 '재미있는 영화 이야기를 하자.', '오늘 밤에 영화를 틀려고 해.', '친구랑 영화 보러 가려고.',
 '애니메이션 영화를 볼 생각이야.', '옛날 영화를 다시 보려고 해.', '영화 한 편을 골라야겠어.',
 '이번 휴일에는 극장에 갈 예정이야.', '오늘 볼 영화가 고민이네.', '집에서 감상할 작품을 고르는 중이야.', '다음에 어떤 영화를 볼지 생각하고 있어.',
 '쉬는 날에 극장 상영작을 하나 보고 싶네.', '저녁에 감상할 영화 한 편을 찾아볼까 해.', '친구와 함께 볼 영화 제목을 고르고 있어.', '오랜만에 집에서 영화 감상을 하려고 해요.']),
 'plant': ('식물을 키우는구나. 어떤 식물이야?', [
 '집에서 식물을 키워.', '화분에 새 모종을 심었어.', '작은 식물을 하나 데려왔어.',
 '키우는 식물에 새잎이 났어.', '화분을 더 큰 것으로 바꾸려고.', '베란다에 식물을 놓았어.',
 '분갈이할 흙을 준비했어.', '화원에서 식물을 샀어.', '집에 있는 화분을 돌보고 있어.',
 '모종을 화분으로 옮겼어.', '식물 줄기가 자라고 있어.', '새로 키울 식물을 골랐어.',
 '방 안에서 화초를 기르고 있어.', '새 화분에 모종을 옮겨 심었네.', '베란다 화초에 잎이 하나 더 생겼어.', '화원에서 데려온 식물을 돌보는 중이야.',
 '내 방 화분에서 작은 잎이 올라오네.', '화초를 넓은 화분으로 옮겨 심으려 해요.', '키우던 모종의 줄기가 전보다 길어졌어.', '집에서 기를 화초 하나를 새로 장만했어.']),
}


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    DEST.mkdir(parents=True)
    splits = {'train': [], 'validation': [], 'test': []}
    manifest, all_q = [], set()
    for topic, (answer, questions) in TOPICS.items():
        assert len(questions) == 20
        for i, question in enumerate(questions):
            assert question not in all_q
            all_q.add(question)
            split = 'train' if i < 12 else 'validation' if i < 16 else 'test'
            record = {'messages': [{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}]}
            splits[split].append(record)
            manifest.append({'topic': topic, 'split': split, 'question': question, 'reference': answer})
    for split, records in splits.items():
        (DEST / f'{split}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    # 매 회차 20개 미학습 질문을 개발 비교용으로 생성합니다.
    questions = [{'id': f'C{i:02d}', 'topic': r['topic'], 'question': r['question'],
                  'criteria': '이해 가능한 짧은 답변으로 주제 유지. 고정 답변 문자열 일치는 보조 지표.'}
                 for i,r in enumerate((r for r in manifest if r['split']=='test'),1)]
    (DEST/'test_questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in questions),encoding='utf-8')
    (DEST/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    probes = [{'id': f"{r['split']}-{i:03d}", 'split': r['split'], 'topic': r['topic'],
               'question': r['question'], 'reference': r['reference'],
               'criteria': '주제 유지와 이해 가능성을 수동 평가. 고정 답변 일치는 보조 지표.'}
              for i,r in enumerate(manifest,1) if r['split'] in ('train','test')]
    (DEST/'probe_questions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probes),encoding='utf-8')
    (DEST/'README.md').write_text('''# 짧은 응답 통제 실험

학습 60개, 검증 20개, 개발 비교 20개. 5개 주제별 동일 개수.
답변은 주제별 고정 문장 1개이며, 질문만 달라진다.
목적은 질문 주제와 답변을 연결하고 짧은 문장을 안정적으로 생성하는지 확인하는 것이다.
자유 대화나 다중 턴 기억 능력 평가가 아니다. 성공해도 일반 챗봇 완성으로 해석하지 않는다.
학습/검증/비교 사이 질문의 정확 중복은 없지만 뜻이 비슷한 질문은 의도적으로 포함했다.
매 회차 보는 test_questions는 독립 최종 시험이 아니라 개발용이다.
이전 600개 문답과 직접 성능 비교하지 않는다. 데이터/답변 길이/과제 난이도가 함께 바뀌었다.
주제 밖 질문 처리와 답변 다양성은 이번 과제에서 다루지 않는다.
원본 600개 데이터와 기존 모델은 변경하지 않는다.
''',encoding='utf-8')
    script = ROOT/'experiments/train_intent_lr_compare.py'
    shutil.copyfile(script, DEST.parents[1]/script.name)
    archive = shutil.make_archive(str(ROOT/'short_dialogue_control_v1_package'),'zip',DEST.parents[1])
    print(archive)


if __name__ == '__main__':
    main()
