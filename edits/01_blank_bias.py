"""Spontaneous speech is under-emitted: deletions run 10-12 per 100 words against 2-3 insertions, and the
blank wins 61% of frames. Lower the blank logit by 1.0, the blank penalty 12d chose on the tune half, now
written into the head instead of applied at decoding time."""

def apply(model, units):
    model.head_bias[0] -= 1.0
