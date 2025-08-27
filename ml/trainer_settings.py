from contextlib import nullcontext

import torch

class TrainerSettings:
    master_process:bool = True
    seed_offset:int = 0
    ddp:bool = False
    ddp_world_size:int = 1

    log_interval:int = 1
    # disable this to prevent saving checkpoints
    allow_save:bool = False

    # should we save a checkpoint after {checkpoint_save_interval} iterations?
    always_save_checkpoint:bool = False
    checkpoint_save_interval:int = 25

    # evaluate accuracy after {eval_interval} iterations
    eval_interval:int = 10
    # estimate accuracy based on {eval_iter} random samples
    eval_iters:int = 50
    # stop when there is no improvement in validation error during {n} eval intervals
    early_stop:int = 200

    eval_only:bool = False
    wandb_log:bool = False

    compile_model:bool = False
    manual_seed:int|None = None

    def __init__(self, **kwargs):
        self.dtype = kwargs.get('dtype', 'float32')
        self.out_dir = kwargs.get('out_dir', './')
        self.model_dir = kwargs.get('model_dir', './models')

        if torch.cuda.is_available():
            self.device_type = 'cuda'
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        elif torch.backends.mps.is_available():
            self.device_type = 'mps'
        else:
            self.device_type = 'cpu'

        # note: float16 data type will automatically use a GradScaler
        self.ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type in ['cpu', 'mps'] else torch.amp.autocast(device_type=self.device_type, dtype=self.ptdtype)

        if self.manual_seed:
            torch.manual_seed(self.manual_seed + self.seed_offset)
            if self.device_type == 'cuda':
                torch.cuda.manual_seed_all(self.seed_offset)
            elif self.device_type == 'mps':
                torch.mps.manual_seed(self.seed_offset)

    def json(self) -> dict:
        return {attr: value for attr, value in self.__dict__.items() if type(value) in [bool, int, float, str]}

