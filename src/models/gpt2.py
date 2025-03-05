import sys
sys.path.append("/scratch/homes/sfan/multi_doge/src")

import datasets
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import transformers
from transformers import (
    TrainingArguments, 
    MODEL_FOR_CAUSAL_LM_MAPPING,
    CONFIG_MAPPING,
    AutoConfig,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.trainer import TRAINER_STATE_NAME

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

import torch
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from contextlib import nullcontext
import huggingface_hub


@dataclass
class CausalLMOutputWithDomainIDs(CausalLMOutputWithCrossAttentions):
    domain_ids: Optional[torch.LongTensor] = None
    log_loss: Optional[torch.FloatTensor] = None # logarithm of scalar loss value
    reference_pertoken_loss: Optional[torch.FloatTensor] = None  # corresponds to uniq_domain_ids
    pertoken_loss: Optional[torch.FloatTensor] = None  # corresponds to uniq_domain_ids
    token_mask: Optional[torch.FloatTensor] = None  # 1.0 for tokens that are not padding
    hidden_states: Optional[torch.FloatTensor] = None  # embeddings before linear + softmax


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default='gpt2',
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default="n_positions=512,n_embd=768,n_layer=12,n_head=12",
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=True,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

def get_model_from_config(model_args:ModelArguments,
                          reweight=False,
                          ref_model_path=None,):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        print("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            print(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            print(f"New config: {config}")
            
    if reweight:
        if ref_model_path is not None:
            return GPTForReweight(config).from_pretrained(ref_model_path), config
        return GPTForReweight(config), config

    return GPT2LMHeadModel(config), config


class GPTForReweight(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)
        self.ignore_index = -100
        # self.loss_fct: compute mean token loss for standard training
        self.loss_fct = CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
        # self.pertoken_loss_fct: compute token loss for proxy model
        self.pertoken_loss_fct = CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)
        
    def _forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        last_token_only: Optional[bool] = False,):
        """
            last_token_only: whether to return the logit for the last token only,
                of shape (batch_size, vocab_size)
        """
        
        # print('max_token_ids: ', torch.max(input_ids))
        # print('min_token_ids: ', torch.min(input_ids))
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        if last_token_only:
            hidden_states = hidden_states[:, -1]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)
        
        lm_logits = self.lm_head(hidden_states)

        CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "hidden_states"])
        if output_hidden_states:
            return CausalLMOutput(logits=lm_logits, hidden_states=hidden_states)
        else:
            return CausalLMOutput(logits=lm_logits, hidden_states=None)

    
    def forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # new params 
        domain_ids: Optional[torch.LongTensor] = None,
        return_pertoken_losses: Optional[bool] = False,
        last_token_only: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions, CausalLMOutputWithDomainIDs]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not return_pertoken_losses:
          # perform standard training
          fwd_output = self._forward(
                      input_ids=input_ids,
                      past_key_values=past_key_values,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,
                      position_ids=position_ids,
                      head_mask=head_mask,
                      inputs_embeds=inputs_embeds,
                      encoder_hidden_states=encoder_hidden_states,
                      encoder_attention_mask=encoder_attention_mask,
                      use_cache=use_cache,
                      output_attentions=output_attentions,
                      output_hidden_states=output_hidden_states,
                      return_dict=return_dict,
                      last_token_only=last_token_only)
          lm_logits = fwd_output.logits

          loss = None
          pertoken_loss = None
          token_mask = None
          if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))  
        else:
          # train proxy model
          fwd_output = self._forward(
                      input_ids=input_ids,
                      past_key_values=past_key_values,
                      attention_mask=attention_mask,
                      token_type_ids=token_type_ids,
                      position_ids=position_ids,
                      head_mask=head_mask,
                      inputs_embeds=inputs_embeds,
                      encoder_hidden_states=encoder_hidden_states,
                      encoder_attention_mask=encoder_attention_mask,
                      use_cache=use_cache,
                      output_attentions=output_attentions,
                      output_hidden_states=output_hidden_states,
                      return_dict=return_dict,
                      last_token_only=last_token_only)
          lm_logits = fwd_output.logits
          
          loss = None
          pertoken_loss = None
          token_mask = None
          if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            ignore_index = -100
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            pertoken_loss = self.pertoken_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            pertoken_loss = pertoken_loss.view(shift_labels.size(0), shift_labels.size(1))
            token_mask = shift_labels.ne(ignore_index).float() # not equal to PAD

            loss = pertoken_loss.sum() / token_mask.sum() 
        if not return_dict:
            output = (lm_logits, None, fwd_output.hidden_states, None, domain_ids, pertoken_loss, token_mask)
            return ((loss,) + output) if loss is not None else output

        log_loss = torch.log(loss)
        out_hidden_states = fwd_output.hidden_states
        return CausalLMOutputWithDomainIDs(
            loss=loss,
            log_loss=log_loss,
            logits=lm_logits,
            past_key_values=None,
            hidden_states=out_hidden_states,
            attentions=None,
            domain_ids=domain_ids,
            pertoken_loss=pertoken_loss,
            token_mask=token_mask)

## test ##
if __name__ == '__main__':
    from data import DataTrainingArguments, get_train_eval_datasets, get_data_collator, interleave_dataloader, individual_dataloader
    from accelerate import Accelerator
    
    accelerator = Accelerator()
    data_config = DataTrainingArguments(dataset='slim_ood-logiqa-piqa-arc_easy-arc_challenge-hellaswag-sciq',
                                        max_train_samples=10000,
                                        max_eval_samples=5000)
    individual_train_ds, individual_tgt_ds, individual_val_ds, domain_config, tokenizer = get_train_eval_datasets(data_config=data_config,
                                                            verbose=True,)
    data_collator=get_data_collator(tokenizer, do_padding=False, max_length=512)
    train_loader = interleave_dataloader(individual_train_ds, domain_config.train_dw,
                          batch_size = 4,
                          num_worker = 0,
                          data_collator=data_collator)
    tgt_loader = interleave_dataloader(individual_tgt_ds, domain_config.tgt_dw,
                          batch_size = 4,
                          num_worker = 0,
                          data_collator=data_collator)
    val_loader = individual_dataloader(individual_val_ds,
                          batch_size = 4,
                          num_worker = 0,
                          data_collator=data_collator)
    
    for train_batch in train_loader:
        print(train_batch)
        break
    
    for val_batch in val_loader:
        print(val_batch)
        break
    
    model_config = ModelArguments()
    rw_model_gpt2, rw_config = get_model_from_config(model_config, reweight=True)
    
    for train_batch in train_loader:
        output = rw_model_gpt2(**train_batch)
        break
    import pdb
    pdb.set_trace()