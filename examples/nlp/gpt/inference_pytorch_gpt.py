from transformers import GPT2Tokenizer, GPT2LMHeadModel
import ptvsd
ptvsd.enable_attach(address =('127.0.0.1', 4000))
ptvsd.wait_for_attach()
tokenizer = GPT2Tokenizer.from_pretrained('./checkpoint/HuggingFace')
model = GPT2LMHeadModel.from_pretrained('./checkpoint/HuggingFace')
text = ['Hello, I am a little',
        "Good morning! Today is my",
        "There is a question about whether",
        "Where can I find the best"]
text = ['Hello, I am a',
        "Good morning! Today is",
        "There is a question about",
        "Where can I find the"]
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
lm_logits = output.logits[:, -1, :]
print('lm_logits:', lm_logits)
new_ids = lm_logits.argmax(dim=-1)
print('new_ids:', new_ids, 'new_tokens:', tokenizer.decode(new_ids))