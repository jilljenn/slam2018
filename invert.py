import numpy as np

with open('fr_en.pred') as f:
    lines = np.array(f.read().splitlines()).astype(np.float64)

with open('fr_en.inv.pred', 'w') as f:
    f.write('\n'.join(map(str, (1 - lines).tolist())))
