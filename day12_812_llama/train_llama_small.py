# config for training a small LLaMA model for testing
# launch as: python train_llama.py config/train_llama_small.py

wandb_log = True
wandb_project = 'llama-small'
wandb_run_name = 'llama-small'


batch_size = 12
block_size = 512  # smaller for testing
gradient_accumulation_steps = 5

# this makes total number of tokens be 30M
max_iters = 1000
lr_decay_iters = 1000

# eval stuff
eval_interval = 100
eval_iters = 20
log_interval = 10

# weight decay
weight_decay = 1e-1

# LLaMA specific settings
learning_rate = 1e-4  # LLaMA uses smaller learning rate
warmup_iters = 100
min_lr = 1e-5

# model architecture - small LLaMA for testing
n_layer = 4
n_head = 8
n_embd = 256
num_key_value_heads = 8
intermediate_size = 512
rms_norm_eps = 1e-6
rope_theta = 10000.0
bias = False  # LLaMA doesn't use bias
dropout = 0.0

# system
device = 'cuda'
dtype = 'bfloat16'
compile = False  