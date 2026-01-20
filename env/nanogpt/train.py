import os
import sys
import uuid
import math
import glob
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch._inductor.config as config
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import wandb

with open(sys.argv[0]) as f:
    code = f.read()

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        y = y / math.sqrt(24)
        return y

    def forward_with_cache(self, x, cache):
        B, T, C = x.size()
        assert T == 1, "forward_with_cache only supports single token input (T=1)"

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if cache is not None:
            k_cache, v_cache = cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        y = y / math.sqrt(24)
        return y, (k, v)

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(rmsnorm(x))
        x = x + self.mlp(rmsnorm(x))
        return x

    def forward_with_cache(self, x, cache):
        attn_out, new_cache = self.attn.forward_with_cache(rmsnorm(x), cache=cache)
        x = x + attn_out
        x = x + self.mlp(rmsnorm(x))
        return x, new_cache

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # initialize the position embedding at std=0.02 to match the scale of the token embedding.
        if isinstance(module, nn.Embedding) and not hasattr(module, 'LLMC_SKIP_INIT'):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_logits=True):
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)
        x = rmsnorm(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
        return optimizer

    def forward_with_cache(self, idx, caches):
        b, t = idx.size()
        assert t == 1, "forward_with_cache only supports single token input (t=1)"

        if caches is not None and len(caches) > 0 and caches[0] is not None:
            past_length = caches[0][0].size(2)
        else:
            past_length = 0
        pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=idx.device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        if caches is None:
            caches = [None] * len(self.transformer.h)

        new_caches = []
        for i, block in enumerate(self.transformer.h):
            x, new_cache = block.forward_with_cache(x, cache=caches[i])
            new_caches.append(new_cache)

        x = rmsnorm(x)
        logits = self.lm_head(x)
        return logits, new_caches

    def forward_safe(self, idx, targets):
        b, t = idx.size()
        caches = None
        total_loss = 0.0
        num_valid_tokens = 0

        for i in range(t):
            logits, caches = self.forward_with_cache(idx[:, i:i+1], caches)
            target = targets[:, i]
            mask = (target != -1)
            if mask.any():
                loss_i = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    ignore_index=-1,
                    reduction='sum'
                )
                total_loss += loss_i
                num_valid_tokens += mask.sum()

        if num_valid_tokens > 0:
            loss = total_loss / num_valid_tokens
        else:
            loss = torch.tensor(float('nan'), device=idx.device)
        return None, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    return ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance current position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

@dataclass
class Hyperparameters:
    # data
    input_bin = "/juice5b/scr5b/nlp/aihinton/fineweb10B/fineweb_train_*.bin"
    input_val_bin = "/juice5b/scr5b/nlp/aihinton/fineweb10B/fineweb_val_*.bin"
    wandb_name = os.environ.get("WANDB_NAME", "nanogpt")
    wandb_project = os.environ.get("WANDB_PROJECT", "nanogpt-training")
    wandb_log = True  # enable wandb logging by default
    model = "d12"

    # optimization
    batch_size = 32 # batch size in tokens
    sequence_length = 1024 # sequence length
    total_batch_size = 262144 # total desired batch size, in units of #tokens
    num_iterations = 26880 # max number of iterations to run; but hard stop after 2h
    learning_rate = 0.0015
    warmup_iters = 256
    weight_decay = 0.1
    grad_clip = 1.0

    # evaluation hyperparameters: DO NOT CHANGE THESE
    val_loss_every = 0 # every how many steps to evaluate val loss? 0 for only at the end
    val_tokens = 10485760 # how many tokens of validation data? it's important to keep this fixed for consistent comparisons

    output_dir = "pylog124m"


if __name__ == "__main__":
    import time
    import tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    args = Hyperparameters()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024
    assert args.model in {"d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = 0 # each process gets the exact same seed
    print(f"using device: {device}")

    # calculate the number of steps to take in the val loop.
    assert args.val_tokens % (B * T * ddp_world_size) == 0
    val_steps = args.val_tokens // (B * T * ddp_world_size)

    tokens_per_fwdbwd = B * T * ddp_world_size
    if ddp_rank == 0:
        print(f"B={B}, T={T}, ddp_world_size={ddp_world_size}")
        print(f"tokens_per_fwdbwd={tokens_per_fwdbwd}")
        print(f"args.total_batch_size={args.total_batch_size}")
    # Adjust batch size to match total_batch_size requirement
    if args.total_batch_size != tokens_per_fwdbwd:
        B = args.total_batch_size // (T * ddp_world_size)
        tokens_per_fwdbwd = B * T * ddp_world_size
        if ddp_rank == 0:
            print(f"Adjusted B to {B} to match total_batch_size")
    assert args.total_batch_size == tokens_per_fwdbwd

    # set up a context manager following the desired dtype and device
    ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # init the model from scratch
    model_config = {
        "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
        "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
        "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
        "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
    }[args.model]
    model = GPT(model_config)
    model = model.train()#.cuda()
    if hasattr(config, "coordinate_descent_tuning"):
        config.coordinate_descent_tuning = True # suggested by @Chillee
    print0("compiling the model...")
    model = torch.compile(model).cuda()

    # load tokens
    train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    if args.input_val_bin:
        val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
    x, y = train_loader.next_batch()

    # here we wrap model into DDP container
    model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                            learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                            device_type=device)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        assert it <= args.num_iterations
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_iters:
            return args.learning_rate * (it+1) / args.warmup_iters
        # 2) linear decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        assert 0 <= decay_ratio <= 1
        return (0.1 + (1 - decay_ratio)) / (0.1 + 1) * args.learning_rate

    run_id = str(uuid.uuid4())

    # initialize wandb
    if master_process and args.wandb_log:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config={
                "model": args.model,
                "batch_size": args.batch_size,
                "sequence_length": args.sequence_length,
                "total_batch_size": args.total_batch_size,
                "num_iterations": args.num_iterations,
                "learning_rate": args.learning_rate,
                "warmup_iters": args.warmup_iters,
                "weight_decay": args.weight_decay,
                "grad_clip": args.grad_clip,
                "val_loss_every": args.val_loss_every,
                "val_tokens": args.val_tokens,
                "ddp_world_size": ddp_world_size,
                "model_params": sum(p.numel() for p in raw_model.parameters()),
                "run_id": run_id,
            },
            tags=[args.model, f"world_size_{ddp_world_size}"],
        )
        # log model architecture
        wandb.watch(raw_model, log="all", log_freq=1000)

    # create the output directory if it does not exist
    if master_process and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    training_time_ms = 0
    # start the clock
    torch.cuda.synchronize()
    t_start_total = time.time()  # track total elapsed time
    t_val_loss_0 = time.time()
    for step in range(args.num_iterations + 1):
        t0 = time.time()
        last_step = (step == args.num_iterations)

        # check if training has exceeded 1 hour
        # Synchronize this decision across all ranks to prevent desynchronization
        torch.cuda.synchronize()
        elapsed_time_seconds = time.time() - t_start_total

        # Each rank checks if it has exceeded the time limit
        time_limit_exceeded = elapsed_time_seconds > 1500  # 1500 seconds = 25min; DO NOT CHANGE THIS
        # Synchronize the decision across all ranks using all_reduce with MAX
        # This ensures if ANY rank exceeded the time limit, ALL ranks will stop together
        if ddp_world_size > 1:
            from torch.distributed import ReduceOp
            time_limit_tensor = torch.tensor([1.0 if time_limit_exceeded else 0.0], device=device)
            torch.distributed.all_reduce(time_limit_tensor, op=ReduceOp.MAX)
            time_limit_exceeded = time_limit_tensor.item() > 0.5
        if time_limit_exceeded:
            print0(f"Training time limit reached ({elapsed_time_seconds:.0f}s > 1500s). Breaking from training loop.")
            last_step = True

        # once in a while evaluate the validation dataset
        if ((args.val_loss_every > 0 and step % args.val_loss_every == 0) or last_step) \
            and (val_loader is not None):
            # stop the clock
            torch.cuda.synchronize()
            training_time_ms += 1000 * (time.time() - t_val_loss_0)
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                for _ in range(val_steps):
                    x_val, y_val = val_loader.next_batch()
                    _, loss = model.module.forward_safe(x_val, y_val)
                    val_loss += loss.item()
                val_loss /= val_steps
            # log to console
            print0(f"val loss (safe) {val_loss}")
            if master_process:
                print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms')

                # log to wandb
                if args.wandb_log:
                    wandb.log({
                        "val/loss": val_loss,
                        "step": step,
                        "train_time_ms": training_time_ms,
                        "memory_allocated_mb": torch.cuda.memory_allocated() // 1024 // 1024,
                        "memory_reserved_mb": torch.cuda.memory_reserved() // 1024 // 1024,
                    }, step=step)

            # start the clock again
            torch.cuda.synchronize()
            t_val_loss_0 = time.time()

        # bit confusing: we want to make sure to eval on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        if last_step:
            break

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        # forward pass
        with ctx:
            _, loss = model(x, y, return_logits=False)
        # advance the dataset for the next batch
        x, y = train_loader.next_batch()
        # backward pass
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = ddp_world_size * B * T / (t1-t0)
        lossf = loss.item() # keep track of the mean loss
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")

        # log to wandb
        if master_process and args.wandb_log:
            wandb.log({
                "train/loss": lossf,
                "train/grad_norm": norm,
                "train/learning_rate": lr,
                "train/tokens_per_second": tokens_per_second,
                "train/step_time_ms": (t1-t0)*1000,
                "step": step,
            }, step=step)

        # keep track of smooth timings, last 20 iterations
        if step > 0 and step > args.num_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # log final metrics to wandb
    if master_process and args.wandb_log:
        wandb.log({
            "final/avg_step_time_ms": np.mean(timings)*1000,
            "final/peak_memory_mb": torch.cuda.max_memory_allocated() // 1024 // 1024,
            "final/total_steps": step,
        })

    # -------------------------------------------------------------------------

    if master_process:
        log = dict(code=code, args=args.__dict__)
        os.makedirs('logs', exist_ok=True)
        torch.save(log, 'logs/%s.pt' % run_id)

    # finish wandb run
    if master_process and args.wandb_log:
        wandb.finish()

    # -------------------------------------------------------------------------
    # clean up nice
    destroy_process_group()