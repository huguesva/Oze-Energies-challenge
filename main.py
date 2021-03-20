import torch
import numpy as np
from args import make_args
from tcn import TCNTrainer

def main(lkwargs):
    for kwargs in lkwargs:
        print('----------ARGUMENTS----------')
        print(kwargs)
        torch.manual_seed(kwargs['seed'])
        tcn = TCNTrainer(**kwargs)
        tcn.fit()

if __name__ == "__main__":
    try:
        import colored_traceback
        colored_traceback.add_hook()
    except ImportError:
        pass
    main(make_args())