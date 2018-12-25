import sys
import os
import numpy as np
from scipy.stats import sem

for root, dirs, files in os.walk(sys.argv[1]):
    if 'log.txt' in files:
        path = os.path.join(root, 'log.txt')
        print(path)
        accs = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('INFO:root:test accuracy'):
                    accs.append(float(line.split(':')[-1]))
        print(np.mean(accs), sem(accs))
