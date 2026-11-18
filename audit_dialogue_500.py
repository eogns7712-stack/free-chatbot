"""500개 학습 문답의 구조·반복·표현·키워드 분포 검사."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

TOPICS = {
    "인사·대화시작": ["안녕", "반가", "인사", "처음", "오랜만", "하이", "대화", "이야기할"],
    "피로·수면": ["피곤", "지쳤", "지쳐", "졸", "잠", "녹초", "기운", "힘이 없"],
    "산책·운동": ["산책", "걷", "걸었", "강변", "운동", "헬스", "다이어트", "등산"],
    "영화·문화": ["영화", "극장", "넷플", "드라마", "책", "독서", "음악", "공연", "노래", "게임"],
    "식물·원예": ["식물", "화분", "분갈이", "새잎", "키우", "꽃"],
    "직장·학업": ["회사", "출근", "퇴근", "직장", "사수", "상사", "공부", "시험", "학교", "과제", "취업", "면접"],
    "관계·연애": ["친구", "가족", "엄마", "아빠", "부모", "연애", "사랑", "남친", "여친", "짝사랑", "결혼", "소개팅", "이별", "썸"],
    "감정·자기평가": ["기분", "마음", "우울", "불안", "걱정", "외로", "짜증", "스트레스", "행복", "자신감", "자존", "뿌듯"],
    "생활·음식·여행": ["여행", "요리", "먹", "음식", "점심", "저녁", "아침", "카페", "커피", "청소", "정리", "빨래", "집", "신발", "날씨", "비가"],
    "건강·금전": ["아프", "병원", "건강", "약", "보험", "돈", "월급", "대출", "저축", "주식"],
}
RULE_DESCRIPTIONS = {
    "짧은질문": "질문이 6자 미만. 맥락 생략 여부를 읽어서 확인.",
    "짧은답변": "답변이 8자 미만. 적절한 인사·짧은 반응일 수도 있음.",
    "질문지시대상생략": "그냥/그거/뭐/점점 등으로 시작. 앞 맥락이 필요한지 확인.",
    "주제키워드엇갈림": "질문·답변에서 감지한 주제가 겹치지 않음. 실제 의미 오류로 확정하지 않음.",
    "무엇인지모르는질문": "질문이 현재 상황·대상을 충분히 특정하지 않을 수 있음.",
}


def normalize(text):
    return re.sub(r"[^가-힣a-z0-9]", "", text.lower())


def style(text):
    end = re.sub(r"[^가-힣A-Za-z0-9]+$", "", text)
    if re.search(r"(요|니다|세요|죠|십시오)$", end):
        return "존댓말형끝맺음"
    if re.search(r"(어|아|야|해|네|지|자|까|니|줘|래|길|라|다)$", end):
        return "비격식형끝맺음"
    return "미분류"


def topics(text):
    return [topic for topic, words in TOPICS.items() if any(word in text for word in words)]


def audit(data):
    source = data / "train.jsonl"
    records = [json.loads(s) for s in source.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8-sig"))
    origins = {r["train_line"]: r for r in manifest}
    assert len(origins) == len(records)
    entries = []
    source_counts = Counter()
    styles = defaultdict(Counter)
    topic_counts = defaultdict(Counter)
    answers = Counter()
    questions = Counter()
    flags = Counter()

    for number, record in enumerate(records, 1):
        messages = record["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError(f"{number}행: 사용자·챗봇 메시지 두 개가 필요합니다")
        question, answer = [m["content"].strip() for m in messages]
        if not question or not answer:
            raise ValueError(f"{number}행: 빈 문장")
        origin = origins[number]
        assert origin["question"] == question
        group = origin["source"]
        source_counts[group] += 1
        styles[group][style(answer)] += 1
        qtopics, atopics = topics(question), topics(answer)
        for topic in qtopics or ["키워드미분류"]:
            topic_counts[group][topic] += 1
        answers[normalize(answer)] += 1
        questions[normalize(question)] += 1
        reasons = []
        if len(question) < 6:
            reasons.append("짧은질문")
        if len(answer) < 8:
            reasons.append("짧은답변")
        if re.match(r"^(그냥|그거|그게|뭐가|뭐지|점점|이거|이게|그럴)", question):
            reasons.append("질문지시대상생략")
        if qtopics and atopics and not set(qtopics) & set(atopics):
            reasons.append("주제키워드엇갈림")
        if question in ("그냥 할까?", "점점 좋아져", "그런가봐", "그렇겠지"):
            reasons.append("무엇인지모르는질문")
        flags.update(reasons)
        entries.append({
            "train_line": number,
            "source": group,
            "csv_line": origin.get("csv_line"),
            "question": question,
            "answer": answer,
            "question_keyword_topics": qtopics,
            "answer_keyword_topics": atopics,
            "ending_style": style(answer),
            "review_flags": reasons,
            "review": {"decision": "pending", "note": ""},
        })
    summary = {
        "train_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "total": len(records),
        "source_counts": dict(source_counts),
        "normalized_unique_questions": len(questions),
        "normalized_unique_answers": len(answers),
        "duplicate_question_groups": sum(v > 1 for v in questions.values()),
        "ending_styles_by_source": {k: dict(v) for k, v in styles.items()},
        "question_keyword_topics_by_source": {k: dict(v) for k, v in topic_counts.items()},
        "flag_counts": dict(flags),
        "flagged_records": sum(bool(e["review_flags"]) for e in entries),
        "flag_explanations": RULE_DESCRIPTIONS,
        "top_answer_prefixes": Counter(e["answer"][:6] for e in entries).most_common(15),
        "top_exact_answers": Counter(e["answer"] for e in entries).most_common(15),
        "limitations": [
            "주제는 부분 문자열 키워드 감지이며 의미 분석이나 주제의 정답 분류가 아닙니다.",
            "한 문답이 여러 주제에 집계되므로 주제 합계가 문답 수보다 클 수 있습니다.",
            "문장 끝 말투 분류는 근사 규칙이며 실제 높임법 판정과 다를 수 있습니다.",
            "검토 플래그는 제거 지시가 아닙니다. 의미 적절성은 사람이 원문을 읽어 판단해야 합니다.",
        ],
    }
    return summary, entries


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data/intent_expanded")
    parser.add_argument("--output", type=Path, default=root / "data/intent_expanded_audit")
    args = parser.parse_args()
    summary, entries = audit(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "review.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in entries), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
