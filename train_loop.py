"""
This training script runs the train.py script but with a loop for testing
"""

import os
import string
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from data_generation import get_train_data
from better_sample import sample_model
import json

# -----------------------------------------------------------------------------
# default config values

out_dir = 'out-add-math-char'
eval_interval = 500 # keep frequent because we'll overfit
eval_iters = 30
log_interval = 20 # don't print too too often
always_save_checkpoint = True
eval_only = False # if True, script exits right after the first eval
init_from = 'scratch' # 'scratch' or 'resume'
# data
dataset = 'add_math_char'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 128
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 5e-5 # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = max_iters
min_lr = learning_rate / 30
stop_iters = 200 # stop training after this many iterations (for debugging)
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99

grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
# wandb logging
wandb_log = True
new_project = False
# use the date and time as the run name
run_date = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time() - 2 * 60 * 60))  # this is UTC time, so PST would be -8 hours
# don't need the year or seconds
run_date = run_date[5:-3]
if new_project:
    # use date and time
    wandb_project = 'add-math-char-' + run_date
else:
    # use the date only
    wandb_project = 'add-math-char-' + run_date[:5]

# -----------------------------------------------------------------------------
# modified config values
# 'default' 10M
# n_layer = 6
# n_head = 6
# n_embd = 384

# 85.00M (1GB)
#n_layer = 12
#n_head = 12
#n_embd = 768
#dropout = 0.0


# model settings
block_size = 128
batch_size = 512
# 128 x 256 = 32,768

# 1.19M
# n_layer = 6
# n_head = 4
# n_embd = 128
# dropout = 0.0

# 3.36M
# n_layer = 6
# n_head = 6
# n_embd = 216

# 3.36M
# n_layer = 6
# n_head = 6 * 2
# n_embd = 216

# 5.49M
n_layer = 6
n_head = 6
n_embd = 276
dropout = 0.0

# 13.45M
# n_layer = 6
# n_head = 6
# n_embd = 216 * 2
# dropout = 0.0

# 20.17M (worse than 13.45M above? - need further testing)
# n_layer = 9
# n_head = 9
# n_embd = 216 * 2

# 6.72M
# n_layer = 3
# n_head = 3
# n_embd = 216 * 2

# 2.24M
# n_layer = 1
# n_head = 1
# n_embd = 216 * 2

# 26.89M (6.72M * 4)
# n_layer = 3
# n_head = 3
# n_embd = 216 * 4

# 6.73M
# n_layer = 6 * 2
# n_head = 6
# n_embd = 216


# run settings
warmup_iters = 200
stop_iters = 50000
eval_interval = stop_iters // 5 # eval 5 times
eval_iters = 30
max_iters = stop_iters
lr_decay_iters = max_iters
log_interval = 20
run_name = f"test-{n_embd}n_embd-{n_head}head-{n_layer}layer-{block_size}ctx-"
# run_name = "test-easy_1-1024n_embd-16head-16layer-256-ctx-out-add-math-char-0.0003-lr-4-base"

# init_from = 'resume'
init_from = 'scratch'

### *** ###
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc.
compile = True

# sample settings
temperature = 0.5
operators = ['+']
num_sample_times = 15 # number of times to sample and test the model during training
sample_interval = stop_iters // num_sample_times  # iter interval to sample from the model for testing it
num_samples = 60 # number of samples to generate per test

# choose the base and symbols
# numbers (base 10)
# base_symbols = string.digits  # (0-9)     ### *** ###
# number (base 4)
base_symbols = string.digits[:4]  # (0-3)     ### *** ###
# manually set the base symbols to 0-3 with offset
# 1
# base_symbols = '1230'
# 2
# base_symbols = '2301'
# 3
# base_symbols = '3012'


# letters (base 26)
# base_symbols = string.ascii_lowercase
# base 24
# base_symbols = string.ascii_lowercase[:24]

# for lower and upper case: (base 52)
# base_symbols = string.ascii_letters # (a-z, A-Z)
# numbers + letters (base 62)
# base_symbols = string.digits + string.ascii_letters

# use the first 10 letters of the alphabet
# base_symbols = base_symbols[:10]
# shift the base symbols
# shift_by = 1
# base_symbols = base_symbols[shift_by:] + base_symbols[:shift_by]

# shuffle the base symbols
randomize_base_symbols = True  ### *** ###
base = len(base_symbols)
print(f'base: {base} symbols: {base_symbols}')

num_backups = 2  # number of checkpoint backups to make during training
# -----------------------------------------------------------------------------
# vocab
def save_vocab(chars, out_dir):
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # save the meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    print(f"saved vocab meta at: {out_dir}/meta.pkl")
# -----------------------------------------------------------------------------
# loop over learning rates
# learning_rates = [1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
# learning_rates = [2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 8e-4, 1e-3]

# learning_rates = [2e-3, 1e-3, 3e-4]
learning_rates = [5e-4]   ### *** ###

for learning_rate in learning_rates:
    # -----------------------------------------------------------------------------
    min_lr = learning_rate / 3   ### *** ###
    # model vocab
    # vocab is '\n', ' ', operators, '=', base_symbols
    vocab = ['\n', ' '] + operators + ['='] + list(base_symbols)
    # sort the vocab
    vocab.sort()
    # if the length is not a power of 2, then pad it
    # while not math.log(len(vocab), 2).is_integer():
    #     # get the next power of 2
    #     next_power_of_2 = 2 ** math.ceil(math.log(len(vocab), 2))
    #     print(f'vocab size: {len(vocab)} is not a power of 2, padding to {next_power_of_2}')
    #     # pad the vocab with symbols not in the vocab, starting with 'A'
    #     for i in range(next_power_of_2 - len(vocab)):
    #         vocab.append(chr(ord('A') + i))
    print(f'vocab: {vocab}')
    wandb_run_name = run_name + run_date + '-' + str(base) + '-base'
    if randomize_base_symbols:
        wandb_run_name += '-r'

    out_dir = run_name + 'out-add-math-char-' + str(learning_rate) + '-lr-' + str(base) + '-base'
    # out_dir = "test-easy_1-1024n_embd-16head-16layer-256-ctx-out-add-math-char-0.0003-lr-4-base"
    wandb_log_copy_dict = {
        'run_name': wandb_run_name,
        'iter': [],
        'loss': [],
        'lr': [],
        'mfu': [],
        'acc_iter': [],
        'accuracy': [],
        'frac_skipped': [],
        'frac_off_by_one': [],
        'median_distance': [],
        'mean_distance': [],
        'variance_distance': []
    }
    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    exec(open('configurator.py').read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        assert gradient_accumulation_steps % torch.cuda.device_count() == 0
        gradient_accumulation_steps //= torch.cuda.device_count()
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    # save vocab
    if init_from == 'scratch':
        save_vocab(vocab, out_dir)

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # poor man's data loader
    # data_dir = os.path.join('data', dataset)
    # train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # def get_batch(split):
    #     data = train_data if split == 'train' else val_data
    #     ix = torch.randint(len(data) - block_size, (batch_size,))
    #     x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    #     y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    #     if device_type == 'cuda':
    #         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    #     else:
    #         x, y = x.to(device), y.to(device)
    #     return x, y

    # -----------------------------------------------------------------------------
    # def get_train_data(length, num_samples=1):
    #     # generate random letters
    #     data = []
    #     for i in range(num_samples):
    #         letters = np.random.choice(list('abcdefghijklmnopqrstuvwxyz'), length)
    #         # convert to string
    #         text = ''.join(letters)
    #         data.append(text)
    #     # print(f"generated {num_samples} samples of length {length}")
    #     # load metadata
    #     with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'rb') as f:
    #         meta = pickle.load(f)
    #     stoi = meta['stoi']
    #     # encode text to integers
    #     ids = [[stoi[c] for c in text] for text in data]
    #     # convert to numpy array with int64 data type
    #     ids = np.array(ids, dtype=np.int64)
    #     return ids

    # modified to use generated data during training
    def get_batch(split):
        if split == 'train':
            # generate training data
            if randomize_base_symbols:
                # base_symbols_shuffled = ''.join(random.sample(base_symbols, len(base_symbols)))
                data = get_train_data(block_size + 10, batch_size, base_symbols, operators, out_dir=out_dir, randomize=True) # dimensions: (batch_size, block_size + 1)
            else:
                data = get_train_data(block_size + 10, batch_size, base_symbols, operators, out_dir=out_dir)
            # data = np.array([get_train_data(block_size + 1) for _ in range(batch_size)])
        else:
            # load validation data from file
            # for testing purposes, we use the same generated data as for training
            val_data = get_train_data(block_size + 10, batch_size, base_symbols, operators, out_dir=out_dir)
            data = val_data

        x = torch.from_numpy(data[:, :block_size]).long()
        y = torch.from_numpy(data[:, 1:block_size+1]).long()

        # x = torch.from_numpy(data[:, :block_size]).long()
        # y = torch.from_numpy(data[:, 1:block_size+1]).long()

        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)

        return x, y
    # -----------------------------------------------------------------------------

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the checkpoint folder
    meta_path = os.path.join(out_dir, 'meta.pkl')
    # meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
        print(f"the vocab is {meta['stoi']}")

    # model init
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                      bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume':
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args['block_size'] = block_size # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y) # forward pass (eval mode)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # training loop
    X, Y = get_batch('train') # fetch the very first batch
    t0 = time.time()
    save_interval = max_iters // (num_backups + 1)
    iter_to_save = list(range(save_interval, max_iters, save_interval))
    # remove last iter_to_save (this is very close to max_iters and thus redundant)
    iter_to_save.pop()
    backup_counter = 0

    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    running_mfu = -1.0
    while True:

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # if wandb_log:
            #     wandb.log({
            #         "iter": iter_num,
            #         "train/loss": losses['train'],
            #         "val/loss": losses['val'],
            #         "lr": lr,
            #         "mfu": running_mfu*100, # convert to percentage
            #     })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt')) # always save to same name (overwriting)

        # save as "ckpt.pt" by default above, but save with a unique name num_backups times during training
        if iter_num in iter_to_save and master_process:
            backup_counter += 1
            print(f"saving backup checkpoint {backup_counter} of {num_backups} to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            if loss.isnan():
                print(f"loss is NaN in iter {iter_num} micro_step {micro_step}")
                continue
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "loss": lossf,
                    "lr": lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
                # save the same metrics to wand_log_copy_dict:
                # (this is an offline copy of the metrics that we can use to plot in the notebook)
                wandb_log_copy_dict['iter'].append(iter_num)
                wandb_log_copy_dict['loss'].append(lossf)
                wandb_log_copy_dict['lr'].append(lr)
                wandb_log_copy_dict['mfu'].append(running_mfu*100)

        if stop_iters is not None and iter_num >= stop_iters:
            print(f"stopping training at {iter_num} iterations")
            best_val_loss = losses['val']
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        # sample the model to test it
        if (iter_num >= stop_iters) or (iter_num > 10 and iter_num % sample_interval == 0 and master_process and not eval_only):
            # save the model to disk unless it was just saved (eval)
            if sample_interval != eval_interval and not (iter_num >= stop_iters):
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
            print(f"sampling model at iter {iter_num}")
            # sample_model(config)
            results = sample_model(config, operators)
            # results is a dictionary:
            # results = {
            #     'accuracy': accuracy,
            #     'frac_skipped': frac_skipped,
            #     'median_distance': med_distance,
            #     'mean_distance': mean_distance,
            #     'variance_distance': var_distance
            # }
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "accuracy": results['accuracy'],
                    "frac_skipped": results['frac_skipped'],
                    "frac_off_by_one": results['frac_off_by_one'],
                    "median_distance": results['median_distance'],
                    "mean_distance": results['mean_distance'],
                    "variance_distance": results['variance_distance'],
                })
                # save the same metrics to wand_log_copy_dict:
                wandb_log_copy_dict['acc_iter'].append(iter_num)
                wandb_log_copy_dict['accuracy'].append(results['accuracy'])
                wandb_log_copy_dict['frac_skipped'].append(results['frac_skipped'])
                wandb_log_copy_dict['frac_off_by_one'].append(results['frac_off_by_one'])
                wandb_log_copy_dict['median_distance'].append(results['median_distance'])
                wandb_log_copy_dict['mean_distance'].append(results['mean_distance'])
                wandb_log_copy_dict['variance_distance'].append(results['variance_distance'])

        # end of iteration
        iter_num += 1
        local_iter_num += 1

        # termination conditions:
        # 1) max_iters
        if iter_num > max_iters:
            break
        # 2) stop_iters
        elif stop_iters is not None and iter_num > stop_iters:
            break

    if ddp:
        destroy_process_group()

    # end wandb run
    if wandb_log and master_process:
        wandb.finish()

    # save plots
    from plot_data import save_plots
    save_plots(wandb_log_copy_dict, config)

    # save the config to disk
    # print(f"saving config to {out_dir}")
    # with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    #     json.dump(config, f)
    # save the wandb log copy dict to disk
    print(f"saving wandb log copy dict to {out_dir}")
    with open(os.path.join(out_dir, 'wandb_log_copy_dict.json'), 'w') as f:
        json.dump(wandb_log_copy_dict, f)  # jason.dump() converts the dict to a string, requires import json
