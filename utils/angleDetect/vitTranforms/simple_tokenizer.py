import regex as re
import ftfy
import html

import os, sys
filePath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(filePath)
#-O 别问，问就是GORK写的，我又没搞过nlp问怎么知道这前面的文本编码部分

def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()

def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

class SimpleTokenizer(object):
    def __init__(self):
        # 词汇表：仅包含 "0" 到 "101"，共 102 个 token
        vocab = [str(i) for i in range(102)]  # ["0", "1", ..., "101"]
        self.encoder = dict(zip(vocab, range(len(vocab))))  # ID 0-101
        self.decoder = {v: k for k, v in self.encoder.items()}
        
        # 禁用 BPE
        self.bpe_ranks = {}
        self.cache = {}
        # 正则表达式：仅匹配数字
        self.pat = re.compile(r"""\d+""")

    def encode(self, text):
        text = whitespace_clean(basic_clean(text)).lower()
        tokens = re.findall(self.pat, text)
        if not tokens:
            return [0]  # 空输入返回默认 ID 0
        # 仅返回第一个匹配的数字 token
        token = tokens[0]
        return [self.encoder.get(token, 0)]  # 未知 token 映射到 ID 0

    def decode(self, tokens):
        text = ''.join([self.decoder.get(token, '') for token in tokens])
        return text