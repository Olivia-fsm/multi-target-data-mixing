from dataclasses import dataclass, field
from typing import List, TypedDict

import sys
sys.path.append("/scratch/homes/sfan/multi_doge/src")
from models import CausalLMOutputWithDomainIDs, ModelArguments, get_model_from_config, GPTForReweight
from schedulers import get_scheduler_extended

import torch


### Original DOGE trainer function
    def train_step_distributed(self, model, inputs):
        self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        loss_all = torch.tensor(0.0)
        effective_domains = 0
        sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
        for i,c in sample_counter.items():
            if i in self.domain_update_counter.keys():
                self.domain_update_counter[i] += c
        
        self.grad_acc_step += 1
        for domain_id in range(len(self.domain_list)):
            new_inputs = self.filter_inputs(inputs, domain_id)
            if new_inputs is None:
                continue
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, new_inputs, return_outputs=True, return_pertoken_losses=False)
            if self.args.world_size>1:
                if self.is_local_process_zero():
                    gathered_losses = [
                            torch.zeros_like(loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(loss, gathered_losses, dst=0)
                    gathered_losses = torch.cat(gathered_losses, dim=0)
                    self.domain_losses_distributed[domain_id] += gathered_losses
                    loss_all += gathered_losses.detach().cpu()
                else:
                    dist.gather(loss, dst=0)
            else:
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                self.domain_losses_distributed[domain_id] = loss+self.domain_losses_distributed[domain_id]
                loss_all += loss.detach().cpu()
            effective_domains += 1
        if self.grad_acc_step == self.args.gradient_accumulation_steps:
            # TODO: update domain weights (DOGE)
            if self.args.gradient_accumulation_steps > 1:
                self.domain_losses_distributed = [l / self.args.gradient_accumulation_steps for l in self.domain_losses_distributed]
            self.update_domain_weights_distributed(self.domain_losses_distributed)

            self.grad_acc_step = 0 
            self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        return loss_all / (self.args.gradient_accumulation_steps*effective_domains)
    
    def update_domain_weights_distributed(self, domain_losses_distributed):
        wandb_log_dict = {}
        full_grad_dicts = []
        for domain_id in range(len(self.domain_list)):
            self.model.zero_grad()
            curr_domain_losses = domain_losses_distributed[domain_id]
            if curr_domain_losses > 0.0:
                if self.use_apex:
                    with amp.scale_loss(curr_domain_losses, self.optimizer) as scaled_curr_domain_losses:
                        scaled_curr_domain_losses.backward()
                else:
                    self.accelerator.backward(curr_domain_losses)
                self.iter_domain_losses[domain_id] = curr_domain_losses.detach().cpu().item()
                # get domain grad
                if self.args.max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                domain_flat_grad = get_model_grad_flat(self.model, tgt_params_ls=self.selected_modules)
                
                self.flat_grad_mat[domain_id][:] = domain_flat_grad
                if domain_id not in self.train_ids:
                    full_grad_dicts.append(None)
                else:
                    domain_full_grad_dict = get_model_grad_dict(self.model)
                    full_grad_dicts.append(domain_full_grad_dict)
            else:
                full_grad_dicts.append(None)
            self.model.zero_grad()
        train_mat = self.flat_grad_mat[self.train_ids][:]
        tgt_mat = self.flat_grad_mat[self.tgt_ids][:]
        scores_mat = train_mat @ tgt_mat.T
        
        lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 1e-4
        
        # TODO: Check dimension 
        if set(self.train_ids) == set(self.tgt_ids):
            scores = lr_t * (scores_mat.sum(dim=-1) - scores_mat.diag())
        else:
            scores = lr_t * scores_mat.sum(dim=-1)
        
        avg_norm = train_mat.norm(dim=-1).mean()
        scores = scores/(avg_norm+1e-6)
        scores = torch.clip(scores, min=lr_t*self.dw_min, max=lr_t*self.dw_max)
        
        dw_prev = self.train_dw
        log_dw_new = torch.log(dw_prev[self.train_ids]) + scores / self.mu
        dw_new = torch.nn.functional.softmax(log_dw_new, dim=-1)
        dw_new = (1-self.reweight_eps) * dw_new + self.reweight_eps / len(dw_new) # default reweight_eps=0.0, no smoothing
        self.train_dw[self.train_ids] = dw_new
        self.avg_dw[self.train_ids] += dw_new
        self.dw_update_steps += 1
        add_model_grad_ls(self.model, [full_grad_dicts[i] for i in self.train_ids], dw=self.train_dw[self.train_ids])
        self.write_weights(cur_weights=self.train_dw, avg_weights=self.avg_dw/self.dw_update_steps)
        
        grad_norm = self.flat_grad_mat.norm(dim=-1)
        for domain_idx in range(len(self.domain_list)):
            domain_name = self.idx2domain[domain_idx]
            if domain_idx in self.train_ids:
                score_idx = self.train_ids.tolist().index(domain_idx)
                wandb_log_dict[f'score/{domain_name}'] = scores[score_idx].item()
            elif domain_idx in self.tgt_ids:
                wandb_log_dict[f'score/{domain_name}'] = 0.0
            wandb_log_dict[f'grad_norm/{domain_name}'] = grad_norm[domain_idx].item()
            wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
            wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
            wandb_log_dict[f'loss/{domain_name}'] = self.iter_domain_losses[domain_idx]
            if domain_idx in self.domain_update_counter.keys():
                wandb_log_dict[f'sample_count/{domain_name}'] = self.domain_update_counter[domain_idx]    
        wandb_log_dict['lr'] = lr_t
        
        wandb.log(wandb_log_dict, commit=False)
    
    def train_step_doremi(self, model, inputs):
        assert self.doremi, "Only run this function for doremi!"
        self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        self.ref_domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        
        loss_all = torch.tensor(0.0)
        ref_loss_all = torch.tensor(0.0)
        effective_domains = 0
        sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
        for i,c in sample_counter.items():
            if i in self.domain_update_counter.keys():
                self.domain_update_counter[i] += c
        # print("diremi counter: ", sample_counter)
        self.grad_acc_step += 1
        for domain_id in range(len(self.domain_list)):
            new_inputs = self.filter_inputs(inputs, domain_id)
            if new_inputs is None:
                continue
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, new_inputs, return_outputs=True, return_pertoken_losses=False)
                ref_loss, ref_outputs = self.compute_loss(self.ref_model, new_inputs, return_outputs=True, return_pertoken_losses=False)
            
            if self.args.world_size>1:
                if self.is_local_process_zero():
                    gathered_losses = [
                            torch.zeros_like(loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(loss, gathered_losses, dst=0)
                    gathered_losses = torch.cat(gathered_losses, dim=0)
                    self.domain_losses_distributed[domain_id] += gathered_losses
                    loss_all += gathered_losses.detach().cpu()
                    
                    ref_gathered_losses = [
                            torch.zeros_like(ref_loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(ref_loss, ref_gathered_losses, dst=0)
                    ref_gathered_losses = torch.cat(ref_gathered_losses, dim=0)
                    self.ref_domain_losses_distributed[domain_id] += ref_gathered_losses
                    ref_loss_all += ref_gathered_losses.detach().cpu()
                else:
                    dist.gather(loss, dst=0)
                    dist.gather(ref_loss, dst=0)
            else:
                if self.args.n_gpu > 1:
                    loss = loss.mean()
                    ref_loss = ref_loss.mean()
                self.domain_losses_distributed[domain_id] = loss+self.domain_losses_distributed[domain_id]
                self.ref_domain_losses_distributed[domain_id] = ref_loss+self.ref_domain_losses_distributed[domain_id]
                ref_loss_all += ref_loss.detach().cpu()
                loss_all += loss.detach().cpu()
            effective_domains += 1
        if self.grad_acc_step == self.args.gradient_accumulation_steps:
            # TODO: update domain weights (DOGE)
            if self.args.gradient_accumulation_steps > 1:
                self.domain_losses_distributed = [l / self.args.gradient_accumulation_steps for l in self.domain_losses_distributed]
                self.ref_domain_losses_distributed = [l / self.args.gradient_accumulation_steps for l in self.ref_domain_losses_distributed]
            self.update_domain_weights_doremi(self.domain_losses_distributed, self.ref_domain_losses_distributed)

            self.grad_acc_step = 0 
            self.domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
            self.ref_domain_losses_distributed = [torch.tensor(0.0) for _ in range(len(self.domain_list))] 
        return loss_all / (self.args.gradient_accumulation_steps*effective_domains)
    
    def update_domain_weights(self, pertoken_losses, token_masks, domain_ids):
        wandb_log_dict = {}
        domain_ids = domain_ids.detach()

        if self.doge:
            full_grad_dicts = []
            all_domain_losses = []
            for domain_id in range(len(self.domain_list)):
                self.model.zero_grad()
                domain_mask = (domain_ids == domain_id)
                # import pdb
                # pdb.set_trace()
                if domain_mask.sum() > 0:
                    curr_domain_losses = pertoken_losses[token_masks*domain_mask.reshape(-1, 1)].mean()
                    all_domain_losses.append(curr_domain_losses)
                else:
                    all_domain_losses.append(None)
            
            for domain_id, curr_domain_losses in enumerate(all_domain_losses):
                if curr_domain_losses is None:
                    full_grad_dicts.append(None)
                else:
                    if self.use_apex:
                        with amp.scale_loss(curr_domain_losses, self.optimizer) as scaled_curr_domain_losses:
                            scaled_curr_domain_losses.backward()
                    else:
                        self.accelerator.backward(curr_domain_losses,retain_graph=True)
                    self.iter_domain_losses[domain_id] = curr_domain_losses.detach().cpu().item()
            
                    # get domain grad
                    if self.args.max_grad_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    domain_flat_grad = get_model_grad_flat(self.model, tgt_params_ls=self.selected_modules)
                    domain_full_grad_dict = get_model_grad_dict(self.model)
                    self.flat_grad_mat[domain_id][:] = domain_flat_grad
                    full_grad_dicts.append(domain_full_grad_dict)
            train_mat = self.flat_grad_mat[self.train_ids][:]
            tgt_mat = self.flat_grad_mat[self.tgt_ids][:]
            scores_mat = train_mat @ tgt_mat.T
            
            lr_t = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler is not None else 1e-4
            
            # TODO: Check dimension 
            if set(self.train_ids) == set(self.tgt_ids):
                scores = lr_t * (scores_mat.sum(dim=-1) - scores_mat.diag())
            else:
                scores = lr_t * scores_mat.sum(dim=-1)
            
            avg_norm = train_mat.norm(dim=-1).mean()
            scores = scores/(avg_norm+1e-6)
            scores = torch.clip(scores, min=lr_t*self.dw_min, max=lr_t*self.dw_max)
            
            dw_prev = self.train_dw
            log_dw_new = torch.log(dw_prev[self.train_ids]) + scores / self.mu
            dw_new = torch.nn.functional.softmax(log_dw_new, dim=-1)
            dw_new = (1-self.reweight_eps) * dw_new + self.reweight_eps / len(dw_new) # default reweight_eps=0.0, no smoothing
            self.train_dw[self.train_ids] = dw_new
            self.avg_dw[self.train_ids] += dw_new
            self.dw_update_steps += 1
            add_model_grad_ls(self.model, [full_grad_dicts[i] for i in self.train_ids], dw=self.train_dw[self.train_ids])
            self.write_weights(cur_weights=self.train_dw, avg_weights=self.avg_dw/self.dw_update_steps)
        else:
            raise ValueError(f"Reweighting Scheme not supported")
        grad_norm = self.flat_grad_mat.norm(dim=-1)
        for domain_idx in range(len(self.domain_list)):
            domain_name = self.idx2domain[domain_idx]
            if domain_idx in self.train_ids:
                wandb_log_dict[f'score/{domain_name}'] = scores[domain_idx].item()
            elif domain_idx in self.tgt_ids:
                wandb_log_dict[f'score/{domain_name}'] = 0.0
            wandb_log_dict[f'grad_norm/{domain_name}'] = max(grad_norm[domain_idx].item(), self.args.max_grad_norm)
            wandb_log_dict[f'avg_dw/{domain_name}'] = self.avg_dw[domain_idx].item() / self.dw_update_steps
            wandb_log_dict[f'cur_dw/{domain_name}'] = self.train_dw[domain_idx].item()
            wandb_log_dict[f'loss/{domain_name}'] = self.iter_domain_losses[domain_idx]
        wandb_log_dict['lr'] = lr_t
        
        # wandb_log_dict['max_domain_id'] = self.train_dw.argmax().item()
        wandb.log(wandb_log_dict, commit=False)
    
    def filter_inputs(self, inputs, domain_id):
        selected_ids = inputs['domain_ids']==domain_id
        if selected_ids.sum()==0:
            return None
        
        new_inputs = {}
        for k,v in inputs.items():
            new_inputs[k] = v[selected_ids.flatten()]
        return new_inputs
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                from packaging import version
                from accelerate import __version__ as accelerate_version
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    from accelerate.data_loader import SeedableRandomSampler
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if  not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        # if self.curriculum is not None:
        #     curriculum_steps = sorted(list(self.curriculum.keys()))
        #     num_train_epochs = len(curriculum_steps)
        #     assert epochs_trained==0
        #     for epoch in range(num_train_epochs):
        #         if epoch==len(curriculum_steps)-1:
        #             steps_in_epoch = (args.max_steps-curriculum_steps[epoch]) * args.gradient_accumulation_steps
        #         else:
        #             steps_in_epoch = (curriculum_steps[epoch+1]-curriculum_steps[epoch]) * args.gradient_accumulation_steps
        
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
            # while self.state.global_step < steps_in_epoch:
                # inputs = next(epoch_iterator.__iter__())
                # step += 1
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
    
                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    
    def _inner_training_loop_curriculum(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                from packaging import version
                from accelerate import __version__ as accelerate_version
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    from accelerate.data_loader import SeedableRandomSampler
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        # if self.curriculum is not None:
        #     curriculum_steps = sorted(list(self.curriculum.keys()))
        #     num_train_epochs = len(curriculum_steps)
        #     assert epochs_trained==0
        #     for epoch in range(num_train_epochs):
        #         if epoch==len(curriculum_steps)-1:
        #             steps_in_epoch = (args.max_steps-curriculum_steps[epoch]) * args.gradient_accumulation_steps
        #         else:
        #             steps_in_epoch = (curriculum_steps[epoch+1]-curriculum_steps[epoch]) * args.gradient_accumulation_steps
        
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            # global_step = 0
            while self.state.global_step < steps_in_epoch:
                epoch_iterator = train_dataloader
                    
                for step, inputs in enumerate(epoch_iterator):
                # while self.state.global_step < steps_in_epoch:
                    # inputs = next(epoch_iterator.__iter__())
                    # step += 1
                    total_batched_samples += 1

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                        
                    with self.accelerator.accumulate(model):
                        tr_loss_step = self.training_step(model, inputs)

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        tr_loss += tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    is_last_step_and_steps_less_than_grad_acc = (
                        steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                    )

                    if (
                        total_batched_samples % args.gradient_accumulation_steps == 0
                        or
                        # last step in epoch but step is always smaller than gradient_accumulation_steps
                        is_last_step_and_steps_less_than_grad_acc
                    ):
                        # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                        # in accelerate. So, explicitly enable sync gradients to True in that case.
                        if is_last_step_and_steps_less_than_grad_acc:
                            self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            # deepspeed does its own clipping

                            if is_sagemaker_mp_enabled() and args.fp16:
                                self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                        # Optimizer step
                        self.optimizer.step()
                        optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                        if optimizer_was_run:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                        self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        break
                    
                    if (self.curriculum is not None) and (self.state.global_step>0) and (self.state.global_step in self.curriculum.keys()):
                        new_sample_dw = torch.tensor(self.curriculum[self.state.global_step], dtype=torch.float)
                        train_dataloader.dataset._ex_iterable.probabilities_handle = new_sample_dw
                        train_dataloader.dataset._ex_iterable.probabilities = new_sample_dw
                        print(f"Current Training Domain Weights (step={self.state.global_step}):", new_sample_dw)
                    
                        # import pdb
                        # pdb.set_trace()
                        break
                    
                if step < 0:
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_tpu_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        if self.doge:
            if not self.compute_pertoken_losses:
                return self.train_step_distributed(model, inputs)
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True, return_pertoken_losses=True)
                pertoken_loss = outputs.pertoken_loss
                token_mask = outputs.token_mask
                # reference_pertoken_loss = outputs.reference_pertoken_loss
                # excess_loss = pertoken_loss - reference_pertoken_loss

            # print(self.args.world_size)
            if self.args.world_size>1:
                if self.is_local_process_zero():
                    gathered_pertoken_losses = [
                            torch.zeros_like(pertoken_loss) for _ in range(self.args.world_size)
                            ]
                    dist.gather(pertoken_loss, gathered_pertoken_losses, dst=0)
                    gathered_pertoken_losses = torch.cat(gathered_pertoken_losses, dim=0)

                    gathered_token_mask = [
                            torch.zeros_like(token_mask) for _ in range(self.args.world_size)
                            ]
                    dist.gather(token_mask, gathered_token_mask, dst=0)
                    gathered_token_mask = torch.cat(gathered_token_mask, dim=0)

                    gathered_domain_id = [
                            torch.zeros_like(inputs['domain_ids']) for _ in range(self.args.world_size)
                            ]
                    dist.gather(inputs['domain_ids'], gathered_domain_id, dst=0)
                    gathered_domain_id = torch.cat(gathered_domain_id, dim=0)

                    self.pertoken_losses_all.append(gathered_pertoken_losses)
                    self.token_masks.append(gathered_token_mask.detach())
                    self.domain_ids.append(gathered_domain_id.detach())

                    if len(self.pertoken_losses_all) == self.args.gradient_accumulation_steps:
                        pertoken_losses_all = torch.cat(self.pertoken_losses_all, dim=0)
                        token_masks = torch.cat(self.token_masks, dim=0).bool()
                        domain_ids = torch.cat(self.domain_ids, dim=0)

                        # TODO: update domain weights (DOGE)
                        if self.args.gradient_accumulation_steps > 1:
                            pertoken_losses_all = pertoken_losses_all / self.args.gradient_accumulation_steps
                        self.update_domain_weights(pertoken_losses_all, token_masks, domain_ids)

                        self.pertoken_losses_all = []
                        self.token_masks = []
                        self.domain_ids = []
                else:
                    dist.gather(pertoken_loss, dst=0)
                    dist.gather(token_mask, dst=0)
                    dist.gather(inputs['domain_ids'], dst=0)
            else:
                self.pertoken_losses_all.append(pertoken_loss)
                self.token_masks.append(token_mask.detach())
                self.domain_ids.append(inputs['domain_ids'].detach())

                if len(self.pertoken_losses_all) == self.args.gradient_accumulation_steps:
                    pertoken_losses_all = torch.cat(self.pertoken_losses_all, dim=0)
                    token_masks = torch.cat(self.token_masks, dim=0).bool()
                    domain_ids = torch.cat(self.domain_ids, dim=0)

                    # TODO: update domain weights (DOGE)
                    if self.args.gradient_accumulation_steps > 1:
                        pertoken_losses_all = pertoken_losses_all / self.args.gradient_accumulation_steps
                    self.update_domain_weights(pertoken_losses_all, token_masks, domain_ids)

                    self.pertoken_losses_all = []
                    self.token_masks = []
                    self.domain_ids = []
            return loss.detach() / self.args.gradient_accumulation_steps
        else:
            wandb_log_dict = {}
            sample_counter = Counter(inputs["domain_ids"].flatten().detach().cpu().numpy())
            for i,c in sample_counter.items():
                if i in self.domain_update_counter.keys():
                    self.domain_update_counter[i] += c
            for domain_idx in self.domain_update_counter.keys():
                domain_name = self.idx2domain[domain_idx]
                wandb_log_dict[f'sample_count/{domain_name}'] = self.domain_update_counter[domain_idx]    
            wandb.log(wandb_log_dict, commit=False)
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False, return_pertoken_losses=False)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            return loss.detach()