"""Deleted "så" and "og" are suppressed late in the reader (logit lens: kept and deleted split from block 3 on).
Six feed-forward units in reader block 4 push toward blank by 0.2-0.3 at deleted så/og but almost not at kept
ones (<= 0.1) or on average frames (|push| <= 0.05): ff1 units 2399, 1425, 841 and ff2 units 183, 1666, 607
(direct blank-vs-rest readout, tune half). Cut their outgoing weights."""

UNITS = {"ff1": [2399, 1425, 841], "ff2": [183, 1666, 607]}

def apply(model, units):
    for name, cols in UNITS.items():
        getattr(model.blocks[4], name).net[4].weight[:, cols] = 0
