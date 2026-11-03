"""Edit 03 showed the block-4 suppression units act causally but weakly: zeroing six of them cut så/og deletions
from 612 to 589 and turned only 5 kept tokens into deletions. They are nearly silent elsewhere, so flip them
instead of zeroing, and take every unit with the same profile (deleted-minus-kept blank push > 0.08, |push| on
kept < 0.05 and on all frames < 0.02; tune half): 9 ff1 and 8 ff2 units in block 4, 2 ff1 units in block 5."""

UNITS = {(4, "ff1"): [302, 436, 593, 829, 841, 1425, 1737, 1985, 2399],
         (4, "ff2"): [345, 607, 931, 1505, 1603, 1666, 2259, 2935],
         (5, "ff1"): [143, 1748]}

def apply(model, units):
    for (block, name), cols in UNITS.items():
        getattr(model.blocks[block], name).net[4].weight[:, cols] *= -1
