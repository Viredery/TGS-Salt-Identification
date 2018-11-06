# -*- coding: utf-8 -*-

import random
import numpy as np
import torch
import os


if 1:
    SEED = 1337
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print('set random seed')
    print('\tSEED=%d' % SEED)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    print('set cuda environment')
    print('\ttorch.__version__              =', torch.__version__)
    print('\ttorch.version.cuda             =', torch.version.cuda)
    print('\ttorch.backends.cudnn.version() =', torch.backends.cudnn.version())
    try:
        print('\tos[\'CUDA_VISIBLE_DEVICES\']     =', os.environ['CUDA_VISIBLE_DEVICES'])
        NUM_CUDA_DEVICES = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    except Exception:
        print('\tos[\'CUDA_VISIBLE_DEVICES\']     =', 'None')
        NUM_CUDA_DEVICES = 1

    print('\ttorch.cuda.device_count()      =', torch.cuda.device_count())
    print('\ttorch.cuda.current_device()    =', torch.cuda.current_device())