# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import transformers

# from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
# from fairscale.optim.grad_scaler import ShardedGradScaler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import src.data
import src.evaluation
import src.model
import src.slurm
import src.util
import wandb
from src.options import Options

# scaler = torch.cuda.amp.GradScaler()
# scaler = ShardedGradScaler()


def log_wandb(args, metrics: Dict, step: int = None):
    if args.is_main:
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)


def train(
    model,
    optimizer,
    scheduler,
    step,
    train_dataset,
    eval_dataset,
    opt,
    collator,
    best_dev_em,
    checkpoint_path,
):

    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
        except:
            tb_logger = None
            logger.warning("Tensorboard is not available.")

    # different seed for different sampling depending on global_rank
    torch.manual_seed(opt.global_rank + opt.seed)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )

    if opt.fp16 or opt.bf16:
        scaler = torch.cuda.amp.GradScaler()
    if opt.ampere:
        torch.backends.cuda.matmul.allow_tf32 = True

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    global_step = 0
    model.train()
    with tqdm(
        total=opt.total_steps, desc="Total steps", unit="step", initial=step, smoothing=1
    ) as t:
        while global_step < opt.total_steps:
            epoch += 1
            for i, batch in enumerate(train_dataloader):
                step += 1
                batch = [x.to(opt.global_rank) for x in batch]
                (idx, labels, _, context_ids, context_mask) = batch

                if opt.fp16 or opt.bf16:
                    with torch.cuda.amp.autocast(
                        dtype=torch.bfloat16 if opt.bf16 else torch.float16
                    ):
                        outputs = model(
                            input_ids=context_ids.cuda(),
                            attention_mask=context_mask.cuda(),
                            labels=labels.cuda(),
                        )
                    train_loss = scaler.scale(outputs.loss)
                    train_loss.backward()
                else:
                    outputs = model(
                        input_ids=context_ids, attention_mask=context_mask, labels=labels
                    )
                    train_loss = outputs.loss
                    train_loss.backward()

                # weight update point (reached accumulation steps)
                if step % opt.accumulation_steps == 0:
                    if opt.fp16 or opt.bf16:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
                    if opt.fp16 or opt.bf16:
                        scaler.step(optimizer)
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    if opt.fp16 or opt.bf16:
                        scaler.update()

                    # update progress bar
                    t.set_description(
                        f"(loss={train_loss.item()}, lr={scheduler.get_last_lr()[0]}, gs={global_step})"
                    )
                    t.update()

                    # update wandb
                    metrics = {
                        "train/loss": train_loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                    }
                    log_wandb(opt, metrics, global_step)
                    
                    global_step += 1

                train_loss = src.util.average_main(train_loss, opt)
                curr_loss += train_loss.item()

                if (
                    global_step % opt.eval_freq == 0
                    and global_step > 0
                    and step % opt.accumulation_steps == 0
                ):
                    logger.info(f"at global step = {global_step} - running evaluation")
                    dev_em = evaluate(model, eval_dataset, tokenizer, collator, opt)
                    model.train()
                    if opt.is_main:
                        if dev_em > best_dev_em:
                            best_dev_em = dev_em
                            src.util.save(
                                model,
                                optimizer,
                                scheduler,
                                global_step,
                                best_dev_em,
                                opt,
                                checkpoint_path,
                                "best_dev",
                            )
                        log = f"{global_step} / {opt.total_steps} |"
                        log += f"train: {curr_loss/opt.eval_freq:.3f} |"
                        log += f"evaluation: {100*dev_em:.2f}EM |"
                        log += f"lr: {scheduler.get_last_lr()[0]:f}"
                        logger.info(log)
                        log_wandb(opt, {"evaluation": 100 * dev_em}, global_step)
                        if tb_logger is not None:
                            tb_logger.add_scalar("Evaluation", dev_em, global_step)
                            tb_logger.add_scalar(
                                "Training", curr_loss / (opt.eval_freq), global_step
                            )
                        curr_loss = 0.0

                if global_step % opt.save_freq == 0 and step > 0:
                    if opt.is_main:
                        src.util.save(
                            model,
                            optimizer,
                            scheduler,
                            global_step,
                            best_dev_em,
                            opt,
                            checkpoint_path,
                            f"step-{global_step}",
                        )
                if global_step > opt.total_steps:
                    break


def evaluate(model, dataset, tokenizer, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collator,
    )
    model.eval()
    total = 0
    exactmatch = []
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="eval"):
            (idx, _, _, context_ids, context_mask) = batch

            outputs = model.generate(
                input_ids=context_ids.cuda(), attention_mask=context_mask.cuda(), max_length=50
            )

            for k, o in enumerate(outputs):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])["answers"]
                score = src.evaluation.ems(ans, gold)
                total += 1
                exactmatch.append(score)

    exactmatch, total = src.util.weighted_average(np.mean(exactmatch), total, opt)
    return exactmatch


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    opt = options.parse()

    import random

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger = src.util.init_logger(opt.is_main, opt.is_distributed, checkpoint_path / "run.log")
    logger.info(options.print_options(opt))

    if opt.model_name is not None:
        model_name = opt.model_name
    else:
        model_name = "t5-" + opt.model_size
    model_class = src.model.FiDT5

    # load data
    # tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
    tokenizer = transformers.T5TokenizerFast.from_pretrained(model_name)
    collator = src.data.Collator(
        opt.text_maxlength,
        tokenizer,
        answer_maxlength=opt.answer_maxlength,
        pad_to_max_length=opt.pad_to_max_length,
    )
    # use golbal rank and world size to split the eval set on multiple gpus
    train_examples = src.data.load_data(
        opt.train_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    train_dataset = src.data.Dataset(train_examples, opt.n_context)
    # use golbal rank and world size to split the eval set on multiple gpus
    eval_examples = src.data.load_data(
        opt.eval_data,
        global_rank=opt.global_rank,
        world_size=opt.world_size,
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    if opt.overwrite_checkpoint and opt.model_path == "none":
        # no checkpoint exists or forced overwrite
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
        model = src.model.FiDT5(t5.config)
        model.load_t5(t5.state_dict())
        model = model.to(opt.global_rank)
        optimizer, scheduler = src.util.set_optim(opt, model)
        step, best_dev_em = 0, 0.0
    elif opt.model_path == "none":
        # load checkpoint
        load_path = checkpoint_path / "checkpoint" / "latest"
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = src.util.load(
            model_class, load_path, opt, reset_params=False
        )
        logger.info(f"Model loaded from {load_path}")
    else:
        model, optimizer, scheduler, opt_checkpoint, step, best_dev_em = src.util.load(
            model_class, opt.model_path, opt, reset_params=True
        )
        logger.info(f"Model loaded from {opt.model_path}")

    model.set_checkpoint(opt.use_checkpoint)

    if opt.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

    if opt.is_main:
        wandb.init(project="fid", config=opt)
    logger.info("Start training")
    train(
        model,
        optimizer,
        scheduler,
        step,
        train_dataset,
        eval_dataset,
        opt,
        collator,
        best_dev_em,
        checkpoint_path,
    )
