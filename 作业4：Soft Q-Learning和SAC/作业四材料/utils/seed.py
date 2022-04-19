import torch
import numpy as np
import random
import os

# call seed functions
# seednum: seeding number (default - 1)
def seed(seednum = 1):
    random.seed(seednum)
    np.random.seed(seednum)
    torch.manual_seed(seednum)
    torch.cuda.manual_seed(seednum)
    os.environ['PYTHONHASHSEED'] = str(seednum)
    torch.backends.cudnn.deterministic = True