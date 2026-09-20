"""Target-disjoint validation slices; history may reuse previously observed values."""
import numpy as np


def validation_masks(origins, horizon, train_end, validation_end):
    boundary = (train_end + validation_end) // 2
    calibration = origins + horizon <= boundary
    decision = origins >= boundary
    assert calibration.any() and decision.any(), "Too few observations for calibration/decision"
    assert origins[calibration].max() + horizon <= origins[decision].min()
    return calibration, decision, boundary


def decision_segments(origins, horizon, boundary, validation_end):
    middle = (boundary + validation_end) // 2
    first = origins + horizon <= middle
    second = origins >= middle
    if first.any() and second.any():
        assert origins[first].max() + horizon <= origins[second].min()
    return first, second, middle


def self_check():
    # Security H12 has enough total decision windows but not two disjoint subperiods.
    origins = np.arange(216, 247 - 12 + 1)
    cal, dec, cut = validation_masks(origins, 12, 216, 247)
    assert (cal.sum(), dec.sum()) == (4, 5)
    assert len(set(j for i in origins[cal] for j in range(i, i+12)) &
               set(j for i in origins[dec] for j in range(i, i+12))) == 0
    first, second, _ = decision_segments(origins[dec], 12, cut, 247)
    assert not first.any() and not second.any()
    origins = np.arange(890, 1017 - 24 + 1)
    cal, dec, cut = validation_masks(origins, 24, 890, 1017)
    first, second, _ = decision_segments(origins[dec], 24, cut, 1017)
    assert first.any() and second.any()
    print("target boundary self-check passed")


if __name__ == '__main__':
    self_check()
