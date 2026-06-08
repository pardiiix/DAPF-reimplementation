# OPTConfig, OPTForCausalLM, \  GPTJConfig, GPTJForCausalLM
from statistics import mode
from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from .utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from .mlm import MLMTokenizerWrapper
from .seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
from .lm import LMTokenizerWrapper
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
    RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM, \
    AlbertTokenizer, AlbertConfig, AlbertModel, AlbertForMaskedLM, \
    T5Config, T5Tokenizer, T5ForConditionalGeneration, \
    OpenAIGPTTokenizer, OpenAIGPTLMHeadModel, OpenAIGPTConfig, \
    GPT2Config, GPT2Tokenizer, GPT2LMHeadModel, \
    ElectraConfig, ElectraForMaskedLM, ElectraTokenizer, \
    DistilBertConfig, DistilBertModel, DistilBertTokenizer, DistilBertForMaskedLM, AutoModelForMaskedLM

from collections import namedtuple
from yacs.config import CfgNode

from openprompt.utils.logging import logger
from openprompt.plms.keyword_extractor import DomainScorer
import nltk
from openprompt.plms.bertmaskedlmsequenceclassification import BertMaskedlmForSequenceClassification,DistilBertMaskedlmForSequenceClassification

from openprompt.plms.sequence_classification import (
    BertPrefixForSequenceClassification,
    BertPromptForSequenceClassification,
    BertPrefixv2ForSequenceClassification
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForTokenClassification,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice
)

ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))

_MODEL_CLASSES = {
    'bertprompt': ModelClass(**{
        'config': AutoConfig,
        'tokenizer': AutoTokenizer,
        'model':BertPromptForSequenceClassification,
        'wrapper': MLMTokenizerWrapper,

    }),
    'bertprefix': ModelClass(**{          #used for switchprompt
        'config': AutoConfig,
        'tokenizer': AutoTokenizer,
        'model': BertPrefixForSequenceClassification,
        'wrapper': MLMTokenizerWrapper,

    }),
    'bertprefixv2': ModelClass(**{
        'config': AutoConfig,
        'tokenizer': AutoTokenizer,
        'model': BertPrefixv2ForSequenceClassification,
        'wrapper': MLMTokenizerWrapper,

    }),
    'bert_baseline': ModelClass(**{
        'config': AutoConfig,
        'tokenizer': AutoTokenizer,
        'model': AutoModelForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    'bio_clinicalbert': ModelClass(**{
            'config': BertConfig,
            'tokenizer': BertTokenizer,
            'model':BertForMaskedLM,
            'wrapper': MLMTokenizerWrapper,
        }),
    'bert': ModelClass(**{
        'config': BertConfig,
        'tokenizer': BertTokenizer,
        'model':BertForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    'roberta': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model':RobertaForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'distilbert': ModelClass(**{
            'config': DistilBertConfig,
            'tokenizer': DistilBertTokenizer,
            'model':DistilBertForMaskedLM,
            'wrapper': MLMTokenizerWrapper,
        }),
    'distilbert_baseline': ModelClass(**{
            'config': DistilBertConfig,
            'tokenizer': DistilBertTokenizer,
            'model': DistilBertModel,
            'wrapper': MLMTokenizerWrapper,
        }),
    'roberta_baseline': ModelClass(**{
        'config': RobertaConfig,
        'tokenizer': RobertaTokenizer,
        'model': RobertaModel,
        'wrapper': MLMTokenizerWrapper,
    }),
    'albert': ModelClass(**{
        'config': AlbertConfig,
        'tokenizer': AlbertTokenizer,
        'model': AlbertForMaskedLM,
        'wrapper': MLMTokenizerWrapper
    }),
    'gpt': ModelClass(**{
        'config': OpenAIGPTConfig,
        'tokenizer': OpenAIGPTTokenizer,
        'model': OpenAIGPTLMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    'gpt2': ModelClass(**{
        'config': GPT2Config,
        'tokenizer': GPT2Tokenizer,
        'model': GPT2LMHeadModel,
        'wrapper': LMTokenizerWrapper
    }),
    't5':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5TokenizerWrapper
    }),
    't5-lm':ModelClass(**{
        'config': T5Config,
        'tokenizer': T5Tokenizer,
        'model': T5ForConditionalGeneration,
        'wrapper': T5LMTokenizerWrapper,
    }),
    # 'opt': ModelClass(**{
    #     'config': OPTConfig,
    #     'tokenizer': GPT2Tokenizer,
    #     'model': OPTForCausalLM,
    #     'wrapper': LMTokenizerWrapper,
    # }),
    'electra': ModelClass(**{
        'config': ElectraConfig,
        'tokenizer': ElectraTokenizer,
        'model': ElectraForMaskedLM,
        'wrapper': MLMTokenizerWrapper,
    }),
    # "gptj": ModelClass(**{
    #     "config": GPTJConfig,
    #     "tokenizer": GPT2Tokenizer,
    #     "model": GPTJForCausalLM,
    #     "wrapper": LMTokenizerWrapper
    # }),
}


def get_model_class(plm_type: str):
    print(plm_type)
    return _MODEL_CLASSES[plm_type]

#switchprompt
def get_model(model_args, model_name, model_path,src_dataset,trg_dataset, fix_bert: bool = False):

    # wrapper = model_class.wrapper

    # if model_args.prefix:
    #     config.hidden_dropout_prob = model_args.hidden_dropout_prob
    #     config.pre_seq_len = model_args.pre_seq_len
    #     config.prefix_projection = model_args.prefix_projection
    #     config.prefix_hidden_size = model_args.prefix_hidden_size
    #
    #     model_class = PREFIX_MODELS[config.model_type][task_type]
    #     model = model_class.from_pretrained(
    #         model_args.model_name_or_path,
    #         config=config,
    #         revision=model_args.model_revision,
    #     )
    if model_args.prompt:
        model_class = get_model_class(plm_type=model_name)
        config = model_class.config.from_pretrained(model_path, num_labels=2)  # download BertModel
        config.pre_seq_len = model_args.soft_token_num
        config.prefix_hidden_size=model_args.prefix_hidden_size

        model = model_class.model.from_pretrained(model_path, config=config)
        tokenizer = model_class.tokenizer.from_pretrained(model_path)
        wrapper = model_class.wrapper

        # model_class = PROMPT_MODELS[config.model_type][task_type]
        # model = model_class.from_pretrained(
        #     model_args.model_name_or_path,
        #     config=config,
        #     revision=model_args.model_revision,
        # )
        model.tokenizer = tokenizer
        model.wrapper = wrapper
    elif model_args.prefix:
        if model_args.prefix:
            model_class = get_model_class(plm_type=model_name)
            config = model_class.config.from_pretrained(model_path, num_labels=2)  # download BertModel
            config.hidden_dropout_prob = model_args.hidden_dropout_prob
            config.pre_seq_len = model_args.soft_token_num
            config.prefix_projection = model_args.prefix_projection
            config.prefix_hidden_size = model_args.prefix_hidden_size

            model = model_class.model.from_pretrained(
                model_path,
                config=config

            )
            tokenizer = model_class.tokenizer.from_pretrained(model_path)
            wrapper = model_class.wrapper

        model.tokenizer = tokenizer
        model.wrapper=wrapper

    # call with:
    def read_questions_file(dataset):
        sentences = [input.text_a for input in dataset]
        # sentences = [input for input in dataset]


        print(f'Found {len(sentences)} sentences')
        return sentences

    general_file_path = src_dataset #src dataset
    dataset_file_path = trg_dataset #trg dataset
    ########scorer calculate domain specific keyword score and rank them to add top k keywords in the prompts and add their embedding############
    t_general = read_questions_file(general_file_path)
    t_clinical = read_questions_file(dataset_file_path)
    # t_general = read_questions_file(["he's a good man.","man is mortal."])
    # t_clinical = read_questions_file(["clinical mortality rate of man is higher than woman.","prescription is nedded."])
    scorer = DomainScorer(t_general, t_clinical, transformer=model.bert, transformer_tokenizer=tokenizer)
    # keywords = scorer.get_keywords(sentence)
    model.num_dynamic_keyword = model_args.num_dynamic_keyword
    model.num_static_keyword = model_args.num_static_keyword
    model.scorer = scorer

    return model
#baseline mlm model
def get_baseline_model(model_name, model_path, load_model_path,specials_to_add = None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model, plm is the bertmaskedlm or robertamaskedlm that we trained on our corpus and saved..
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    plm_class = get_model_class(plm_type = model_name)
    plm_config = plm_class.config.from_pretrained(load_model_path, output_attentions=True)

    plm = plm_class.model.from_pretrained(load_model_path, config=plm_config) #load the finetuned mlm model from local directory
    plm_tokenizer = plm_class.tokenizer.from_pretrained(load_model_path)
    # plm_wrapper = plm_class.wrapper
    print('ekhane')
    print(model_name)
    if 'distilbert_baseline' not in model_name:
        print(model_name)
        model=BertMaskedlmForSequenceClassification(config=plm_config,model=plm,num_hidden_states=0)
    else:
        print(model_name)
        model = DistilBertMaskedlmForSequenceClassification(config=plm_config, model=plm, num_hidden_states=0)
    model.tokenizer=plm_tokenizer
    # model.wrapper=plm_wrapper
    return model
#template based DAPF
def load_plm(model_name, model_path, specials_to_add = None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    model_class = get_model_class(plm_type = model_name)
    model_config = model_class.config.from_pretrained(model_path, output_attentions=True)
    print(model_class)
    # you can change huggingface model_config here
    # if 't5'  in model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in model_name: # add pad token for gpt
        specials_to_add = ["<pad>"]
        # model_config.attn_pdrop = 0.0
        # model_config.resid_pdrop = 0.0
        # model_config.embd_pdrop = 0.0
    model = model_class.model.from_pretrained(model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(model_path)
    wrapper = model_class.wrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)

    if 'opt' in model_name:
        tokenizer.add_bos_token=False
    return model, tokenizer, model_config, wrapper




def load_plm_from_config(config: CfgNode):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`model_config`: The wrapper class of this plm.
    """
    plm_config = config.plm
    model_class = get_model_class(plm_type = plm_config.model_name)
    model_config = model_class.config.from_pretrained(plm_config.model_path)
    # you can change huggingface model_config here
    # if 't5'  in plm_config.model_name: # remove dropout according to PPT~\ref{}
    #     model_config.dropout_rate = 0.0
    if 'gpt' in plm_config.model_name: # add pad token for gpt
        if "<pad>" not in config.plm.specials_to_add:
            config.plm.specials_to_add.append("<pad>")
    model = model_class.model.from_pretrained(plm_config.model_path, config=model_config)
    tokenizer = model_class.tokenizer.from_pretrained(plm_config.model_path)
    wrapper = model_class.wrapper
    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=config.plm.specials_to_add)
    return model, tokenizer, model_config, wrapper

def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer.

    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.

    Returns:
        The resized model, The tokenizer with the added special tokens.

    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
                logger.info("pad token is None, set to id {}".format(tokenizer.pad_token_id))
    return model, tokenizer



