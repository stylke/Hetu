from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from torch.optim import SGD
import ptvsd

# ptvsd.enable_attach(address =('127.0.0.1', 4000))
# ptvsd.wait_for_attach()
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
encoded_input['labels'] = encoded_input['input_ids']
opt = SGD(model.parameters(), lr=1e-6)

for _ in range(1):
        output = model(**encoded_input)
        loss = output.loss
        print('loss:', loss, loss.dtype)
        '''
        lm_logits = output.logits[:, -1, :]
        print('lm_logits:', lm_logits)
        new_ids = lm_logits.argmax(dim=-1)
        print('new_ids:', new_ids, 'new_tokens:', tokenizer.decode(new_ids))
        '''
        opt.zero_grad()
        loss.backward()
        opt.step()
        
torch.save(model.state_dict(), './checkpoint/temp/hf_pytorch_model.bin')