from tokenizers import Tokenizer, models, pre_tokenizers, decoders, AddedToken
from transformers import PreTrainedTokenizerFast
SPECIAL = ['<|pad|>', '<|report|>', '<|end|>', '<|answer|>']

def make_tokenizer(path):
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {s: i for i, s in enumerate(alphabet + SPECIAL)}
    t = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    t.decoder = decoders.ByteLevel()
    t.add_special_tokens([AddedToken(s, special=True, normalized=False) for s in SPECIAL])
    hf = PreTrainedTokenizerFast(tokenizer_object=t, bos_token=SPECIAL[1], eos_token=SPECIAL[2], pad_token=SPECIAL[0], additional_special_tokens=[SPECIAL[3]], model_max_length=512)
    hf.chat_template = "{{ bos_token }}{% for message in messages %}{{ message['content'] }}{% endfor %}<|answer|>"
    hf.save_pretrained(path)
    return hf

def prompt(text):
    return '<|report|>' + text + '<|answer|>'
