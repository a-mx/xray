import re
from collections import Counter
from typing import Optional, List, Dict
class Vocabulary:
    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.stoi: Dict[str, int] = {
            self.pad_token: 0,
            self.bos_token: 1,
            self.eos_token: 2,
            self.unk_token: 3,
        }
        self.itos: Dict[int, str] = {v: k for k, v in self.stoi.items()}

    @property
    def pad_idx(self) -> int:
        return self.stoi[self.pad_token]

    @property
    def bos_idx(self) -> int:
        return self.stoi[self.bos_token]

    @property
    def eos_idx(self) -> int:
        return self.stoi[self.eos_token]

    @property
    def unk_idx(self) -> int:
        return self.stoi[self.unk_token]

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", str(text).lower(), flags=re.UNICODE)

    def build(self, texts: List[str]) -> None:
        counter = Counter()
        for t in texts:
            counter.update(self._tokenize(t))

        for token, freq in counter.items():
            if freq >= self.min_freq and token not in self.stoi:
                idx = len(self.stoi)
                self.stoi[token] = idx
                self.itos[idx] = token

    def encode(self, text: str, max_len: Optional[int] = None) -> List[int]:
        tokens = self._tokenize(text)
        if max_len is not None:
            tokens = tokens[: max(0, max_len - 2)]

        ids = [self.bos_idx]
        ids.extend(self.stoi.get(tok, self.unk_idx) for tok in tokens)
        ids.append(self.eos_idx)
        return ids