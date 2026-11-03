"""Deleted function words are usually heard but written without a word start: where "så" is deleted the head's
top non-blank piece is "å" (not "▁s"), where "og" is deleted it is "g", and West "eg" for "jeg" comes out as "e".
Raise every word-initial piece by 0.5 against its inner counterpart, so short unstressed words get their own
word start instead of being glued to a neighbour."""

def apply(model, units):
    for i, p in enumerate(units.pieces):
        if p.startswith("▁"):
            model.head_bias[i] += 0.5
