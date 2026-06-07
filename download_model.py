# from transformers import TransfoXLTokenizer, TransfoXLModel, T5ForConditionalGeneration,T5Tokenizer,XLNetTokenizer, XLNetModel, RobertaTokenizer, RobertaForMaskedLM,AutoModelForMaskedLM,AutoModelForSeq2SeqLM,AutoTokenizer
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaTokenizer, RobertaForMaskedLM, \
#     AutoModelForMaskedLM, GPT2Tokenizer
from transformers import OPTForCausalLM, GPT2Tokenizer, AutoModelForMaskedLM, AutoTokenizer,RobertaTokenizer, RobertaForMaskedLM

import torch
#t5=seq length can be 1024
#longformer Longformer uses a sliding window attention mechanism to allow for larger input sequences: huggingface.co/allenai/longformer-base-4096 Jul 6, 2022 at 11:58 Add a comment 1 Answer Sorted by: 6 There is no theoretical limit on the
# input length (ie number of tokens for a sentence in NLP) for transformers.
# from transformers import GPT2LMHeadModel, GPT2Tokenize
# from transformers import LongformerModel
# model = LongformerModel.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True)
# model_name="t5-large"
model_name='bert-large-uncased'
# model_name='distilbert/distilbert-base-uncased'
# model_name='xlnet-base-cased'
# model_name="emilyalsentzer/Bio_ClinicalBERT" #auto
from transformers.models.opt import OPTForCausalLM

# model_name="facebook/opt-350m"
# model_name="transfo-xl-wt103"
# tokenizer = TransfoXLTokenizer.from_pretrained(model_name)
# model = TransfoXLModel.from_pretrained(model_name)
# model_name='./model/'+'flan-t5-base'+'/'
# tokenizer = XLNetTokenizer.from_pretrained(model_name)
# model = XLNetModel.from_pretrained(model_name)
#model = GPT2LMHeadModel.from_pretrained(model_name)
#tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)#flan-t5-base
# tokenizer = AutoTokenizer.from_pretrained(model_name) #flan-t5-base
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
# tokenizer = AutoTokenizer.from_pretrained(model_name) #bert-base-uncased
# model = AutoModelForMaskedLM.from_pretrained(model_name) #bert-base-uncased
# model = AutoModelForMaskedLM.from_pretrained(model_name) #distilbert-base
# model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
# tokenizer = GPT2Tokenizer.from_pretrained(model_name)

#
tokenizer = AutoTokenizer.from_pretrained(model_name) #bert-base-uncased
model = AutoModelForMaskedLM.from_pretrained(model_name) #bert-base-uncased

# model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
# outputs = model(input_ids, masked_lm_labels=input_ids)
# loss, prediction_scores = outputs[:2]
if '/' in model_name:
    model_name=model_name.split('/')[1]
# tokenizer.save_pretrained('./model/'+model_name+'/')
# model.save_pretrained('./model/'+model_name+'/')
# model_name='bert-large-uncased'
# tokenizer = RobertaTokenizer.from_pretrained(model_name)
# model = RobertaForMaskedLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name) #bert-base-uncased
# model = AutoModelForMaskedLM.from_pretrained(model_name) #bert-base-uncased
tokenizer.save_pretrained('./model/'+model_name+'/')
model.save_pretrained('./model/'+model_name+'/')
# text="The Carolinas Conversations Collection demonstrates how stories about health and wellbeing can construct a bridge between the cultures of the sciences and the humanities by focusing on questions such as these: How do we use language to construct illness: how do we re-cog-nize our changed selves, our changed bodies, our new identities when we find ourselves changed by age and health? How does our inscribed self show traces not only of one’s own leftover bits of memory and experience but also of the response and reflection from others? How do we change our stories to accommodate what we think our listeners expect, when those hearers are professionals or when they are friends? Looking at selfhood has additional implications for older persons with cognitive impairment: the construct of personal identity is a key point in discussion of the ethics concerning the capacity of the person for decision-making and advance directives."
