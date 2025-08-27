import os
from pathlib import Path

default = {
        "model": {
            # number of layers
            "in_dim": -1,       # will be set dynamically based on the input data
            "out_dim": 2048,
            "dropout": 0.0,
            "bias": False,
        },

        "train": {
            # ** training params **
            "learning_rate": 1e-4,

            # sample (minibatch) size
            "batch_size": 32,

            # upper limit to training iterations
            "max_iters": 200,

            # params of the Adam optimizer
            "beta1": 0.9,
            "beta2": 0.999,

            # attenuate gradients larger than {grad_clip} to shield
            # weight updates from exploding gradients
            "grad_clip": 1.0,

            # should we decay the learning rate during training
            # (e.g. by multiplying it by 0.1 every 100 iterations)
            "decay_lr": True,
            # how many steps to warm the learning rate up for
            "warmup_iters": 0,

            # weight_decay (wd) penalizes the model for the squared sum of weights * wd
            # this helps to prevent overfitting. wd=0 disables weight regularization while
            # wd > 1 amplifies it. usually, wd=0.1.
            # see more about wd here: https://www.fast.ai/posts/2018-07-02-adam-weight-decay.html
            "weight_decay": 0.0,

            # values > 1 lead to the increasing of effective {batch_size}
            # enables to simulate larger batch sizes when memory is limiting
            "gradient_accumulation_steps": 1,
        }
    }

# how many steps to decay the learning rate for
default["train"]["lr_decay_iters"] = default["train"]["max_iters"]
# decay the learning rate to this value
default["train"]["min_lr"] = default["train"]["learning_rate"] / 10

configurations = {
}

# load all datasets that you want to train on from the cache directory
cache_dir = Path(os.getenv('CACHE_DIR', 'cache'))
for subdir in [d for d in cache_dir.iterdir() if d.is_dir()]:
    for files in [
          'concepts.pkl', 
          'propositions.pkl', 
          #'scitail_test.pkl',
          #'entailment_minitest.pkl',
          # a mix of two datasets
          #['propositions.pkl', 'scitail_test.pkl']
        ]:

        # if the file exists in the subdirectory, add it to the configurations
        if isinstance(files, list):
            datapaths = [subdir / f for f in files if os.path.isfile(subdir / f)]
        elif os.path.isfile(subdir / files):
            datapaths = [subdir / files]
        else:
            continue

        if isinstance(files, str):
            file_prefix = files.split('.')[0]
        else:
            file_prefix = '&'.join([f.split('.')[0] for f in files])

        key = f"{file_prefix}/{subdir.name}"
        configurations[key] = {
            **default,
            "data": {
                "dataset": [str(dp) for dp in datapaths],
                "dataset_name": file_prefix,
                "embedding_model": subdir.name,
                # default endpoint
                "endpoint": f"a -> b",
            }
        }
        if len(datapaths) > 1:
            # more data requires more iterations
            max_iters = 1000
            configurations[key]["train"]["max_iters"] = max_iters
            configurations[key]["train"]["lr_decay_iters"] = max_iters
