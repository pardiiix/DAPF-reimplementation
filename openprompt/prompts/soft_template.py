
import os

from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger



from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn

class SoftTemplate(Template):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens.
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take
    the first n_tokens similar to their implementation).
    """
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self,
                 SOURCE_DOMAINS,
                 TARGET_DOMAINS,
                 N_DMX,
                 N_CTX,
                 CSC,
                 da,
                 classnames,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 text: Optional[str] = None,
                 soft_embeds: Optional[torch.FloatTensor] = None,
                 num_tokens: int=20,
                 initialize_from_vocab: Optional[bool] = True,
                 random_range: Optional[float] = 0.5,
                 placeholder_mapping: dict = {'<text_a>':'text_a','<text_b>':'text_b'},
                ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        self.model=model
        self.tokenizer=tokenizer
        self.raw_embedding = model.get_input_embeddings()
        self.raw_embedding.requires_grad_(False)
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab
        self.n_dmx = N_DMX #num. od domain context vector
        self.n_ctx= N_CTX #num of context vector
        self.classnames=classnames
        self.source_domain=SOURCE_DOMAINS
        self.target_domain = TARGET_DOMAINS
        self.n_dm = len(self.source_domain) + len(self.target_domain)  # number of domains each, source and target domains will be in a list
        self.csc=CSC #class specific context
        self.n_cls = len(classnames)
        self.da=da
        self.text = text
        # self.default_text1 = {"placeholder<text_a> <mask>"
        # self.default_text2 = "<text_a> <text_b> <mask>".split()
        # print("num soft token")
        # print(self.num_tokens)
        if soft_embeds is not None:
            self.soft_embeds = soft_embeds
            self.num_tokens = len(soft_embeds)
        else:
            if self.num_tokens>0 and self.n_ctx==0 and self.n_dmx==0:
                self.generate_parameters()
            elif self.n_ctx>0 or self.n_dmx>0: #for DA experiment, add domain specific and class specific context vector
                self.generate_domain_class_context_parameters()



    def on_text_set(self):
        self.text = self.parse_text(self.text)


    def wrap_one_example(self, example) -> List[Dict]:  #TODO this automatic generated template may not be able to process diverse data format.
        if self.text is None:
            logger.warning("You didn't provide text template for softprompt. Using default template, is this intended?")
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)

    def generate_domain_class_context_parameters(self) -> None:
        # """
        #       generate parameters needed for soft tokens embedding in soft-prompt for combining domain and class context
        #       for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        n_cls = len(self.classnames)
        n_ctx = self.n_ctx
        n_dmx = self.n_dmx

        dtype = self.model.dtype
        ctx_dim = self.raw_embedding.weight.size(1)
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        domainnames = self.source_domain + self.target_domain
        domainnames = [
            ", patient's narrative on {}.".format(domain) for domain in domainnames
        ]
        # print(domainnames)
         # number of domain context
        n = n_dmx + n_ctx
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        naive_prompt_prefix = "Patient has diagnosis".replace("_", " ")

        if self.csc:
            print("Initializing class-specific contexts")
            ctx_vectors = torch.FloatTensor(n_cls, n_ctx, ctx_dim).uniform_(-self.random_range, self.random_range)

            # ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype) #class-specific contexts soft embedding
        else:
            print("Initializing a generic context")
            # ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)  #unified
            ctx_vectors = torch.FloatTensor(n_ctx, ctx_dim).uniform_(-self.random_range, self.random_range)


        # nn.init.normal_(ctx_vectors, std=0.02)
        print("ctx vectors size: ".format(ctx_vectors.size()))
        prompt_prefix = " ".join(["X"] * n)

        domain_vectors = torch.FloatTensor(self.n_dm, n_dmx, ctx_dim).uniform_(-self.random_range,
                                                                                     self.random_range)
        # domain_vectors = torch.empty(n_dm, n_dmx, ctx_dim, dtype=dtype)
        # nn.init.normal_(domain_vectors, std=0.02)
        self.ctx_vectors = nn.Parameter(ctx_vectors)  # to be optimized

        self.domain_vectors = nn.Parameter(domain_vectors)

        # print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of domain context words (tokens): {n_dmx}")


        # classnames = [name.replace("_", " ") for name in self.classnames]
        # name_lens = [len(self.tokenizer.encode(name)) for name in classnames]
        # naive_prompts = [
        #     naive_prompt_prefix + " " + name + "." for name in classnames
        # ]
        #
        # prompts = [
        #     prompt_prefix + " " + name + " " + domain + "."
        #     for domain in domainnames for name in classnames
        # ]
        # #
        # temp=[self.tokenizer.tokenize(p) for p in prompts]
        # print(temp)
        # print(type(temp))
        # tokenized_prompts = torch.cat(torch.tensor([self.tokenizer.tokenize(p) for p in prompts][0]))
        # naive_tokenized_prompts = torch.cat([self.tokenizer.tokenize(p) for p in naive_prompts])
        # # tokenized_prompts = torch.stack([self.tokenizer.tokenize(p) for p in prompts])
        # # naive_tokenized_prompts = torch.stack([self.tokenizer.tokenize(p) for p in naive_prompts])
        #
        #
        # with torch.no_grad():
        #     embedding = self.model.token_embedding(tokenized_prompts).type(
        #         dtype)
        #     naive_embedding = self.model.token_embedding(
        #         naive_tokenized_prompts).type(dtype)
        #
        # # These token vectors will be saved when in save_model(),
        # # but they should be ignored in load_model() as we want to use
        # # those computed using the current class names
        # tokenized_prompts = torch.cat(
        #     [tokenized_prompts, naive_tokenized_prompts])
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:,
        #                                      1 + n:, :])  # CLS, EOS


        # self.n_ctx = n_ctx
        # self.csc = cfg.TRAINER.DAPL.CSC
        # self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        # self.name_lens = name_lens
        # self.naive_embedding = naive_embedding.to(
        #     torch.device("cuda"))
    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        if self.initialize_from_vocab:
            soft_embeds = self.raw_embedding.weight[:self.num_tokens].clone().detach()
        else:
            soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)

        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)
        # print('soft embed size')
        # print(self.soft_embeds.size())



    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        if self.num_tokens>0 and self.n_dmx==0 and self.n_ctx==0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 1) #domain agnostic context vector for joint training
            # print('inputs_embeds size')
            # print(inputs_embeds.size())
        elif self.n_dmx>0 and self.n_ctx>0:
            ctx = self.ctx_vectors
            ctx_dim = ctx.size(-1)
            dmx = self.domain_vectors  # dm 16 512
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1)  # dm 16 512
                if not self.csc:
                    ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1,
                                                  -1)  # dm cls 16 512
            else:
                ctx = ctx.unsqueeze(0).expand(self.n_dm, -1, -1,
                                              -1)  # dm cls 16 512

            dmx = dmx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)  # dm cls 16 512
            ctxdmx = torch.cat([ctx, dmx],
                               dim=2).reshape(self.n_cls * self.n_dm,
                                              self.n_ctx + self.n_dmx, ctx_dim) #dimension increase by one

            print('ctxdmx size')
            print(ctxdmx.size())
            print('inputs_embeds size')
            print(self.n_cls)
            print(self.n_dm)
            inputs_embeds=inputs_embeds.unsqueeze(1).expand(-1, self.n_cls * self.n_dm, -1, -1)
            print(inputs_embeds.size())
            print('ctxdmx size')
            print(ctxdmx.size())

            soft_embeds = ctxdmx.repeat(batch_size, 1, 1, 1)
            # self.soft_embeds=soft_embeds=ctxdmx
            print('soft_embeds size')
            print(soft_embeds.size())
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds], 2)


            # prefix = self.token_prefix
            # suffix = self.token_suffix

            # naive
            # neb = self.naive_embedding

            # prompts = torch.cat(
            #     [
            #         prefix,  # (n_cls, 1, dim)
            #         ctxdmx,  # (n_cls, n_ctx, dim)
            #         suffix,  # (n_cls, *, dim)
            #     ],
            #     dim=1,
            # )
            # # prompts = torch.cat([prompts, neb], dim=0)


        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens>0 and  self.n_dmx==0 and self.n_ctx==0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_tokens), dtype = am.dtype,device=am.device), am], dim=-1)
        if 'attention_mask' in batch and self.n_dmx>0 and self.n_ctx>0:
            am = batch['attention_mask'].unsqueeze(1).expand(-1, self.n_cls * self.n_dm, -1)
            # print(am.size())
            # print(am)
            batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.n_cls * self.n_dm,self.n_ctx + self.n_dmx), dtype = am.dtype,device=am.device), am], dim=-1)
        return batch


    def post_processing_outputs(self, outputs: torch.Tensor):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        temp_output=outputs
        if not self.model_is_encoder_decoder :
            if self.num_tokens>0 and  self.n_dmx==0 and self.n_ctx==0:
                temp_output.logits=outputs.logits[:, :self.num_tokens,: ] #logits on soft_tokens
                outputs.logits = outputs.logits[:, self.num_tokens:,: ]
            elif self.n_dmx>0 and self.n_ctx>0:
                outputs.logits = outputs.logits[:,:, self.n_ctx + self.n_dmx:,: ]

        # print('soft embed logits')
        # print(len(temp_output))
        # print(len(temp_output[0]))
        # print(len(temp_output[0][0]))
        # print(len(temp_output[0][0][0])) #size of vocab
        # print(outputs.shape)


        # print(temp_output.size())
        return outputs,temp_output
