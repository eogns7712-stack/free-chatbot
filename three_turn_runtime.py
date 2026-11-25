"""직접 학습 모델의 역할 토큰 직렬화와 최대 3턴 추론."""
from pathlib import Path
from threading import Lock
import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel


class ThreeTurnModel:
    def __init__(self, folder):
        folder = Path(folder)
        torch.set_num_threads(4)
        self.tok = Tokenizer.from_file(str(folder/'tokenizer.json'))
        self.model = GPT2LMHeadModel.from_pretrained(str(folder), local_files_only=True).eval()
        self.ids = {role:self.tok.token_to_id(f'<{role}>') for role in ('user','assistant','eos','pad')}
        if any(v is None for v in self.ids.values()) or self.model.config.vocab_size != self.tok.get_vocab_size():
            raise ValueError('모델과 토크나이저 설정을 확인하세요.')
        self.lock = Lock()

    def prompt_ids(self, messages):
        if [m['role'] for m in messages] not in (['user'], ['user','assistant','user'], ['user','assistant','user','assistant','user']):
            raise ValueError('현재 입력과 직전 문답 두 쌍만 전달할 수 있습니다.')
        prompt=[]
        for m in messages:
            prompt += [self.ids[m['role']], *self.tok.encode(m['content']).ids, self.ids['eos']]
        return prompt + [self.ids['assistant']]

    def generate(self, messages):
        prompt=self.prompt_ids(messages)
        if len(prompt)+48 > self.model.config.n_positions:
            raise ValueError('대화가 길어요. 새 대화를 시작하거나 짧게 입력해 주세요.')
        x=torch.tensor([prompt])
        with self.lock, torch.inference_mode():
            out=self.model.generate(input_ids=x,attention_mask=torch.ones_like(x),max_new_tokens=48,
                                    do_sample=False,num_beams=1,repetition_penalty=1.0,
                                    eos_token_id=self.ids['eos'],pad_token_id=self.ids['pad'])
        return self.tok.decode(out[0,len(prompt):].tolist(),skip_special_tokens=True).strip()


def encode_record(record, tok, final_only=False):
    messages=record['messages']
    if len(messages) not in (2,4,6) or [m['role'] for m in messages] != ['user','assistant']*(len(messages)//2):
        raise ValueError('최대 3턴 문답 형식이 아닙니다.')
    ids,labels=[],[]
    for i,m in enumerate(messages):
        turn=[tok.token_to_id(f"<{m['role']}>"),*tok.encode(m['content']).ids,tok.token_to_id('<eos>')]
        ids.extend(turn)
        score=m['role']=='assistant' and (not final_only or i==len(messages)-1)
        labels.extend([-100,*turn[1:]] if score else [-100]*len(turn))
    if len(ids)>128:
        raise ValueError('학습 문맥 128토큰을 초과합니다.')
    return torch.tensor(ids),torch.tensor(labels)
