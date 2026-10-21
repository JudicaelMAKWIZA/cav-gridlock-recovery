"""Mêmes calculs pour la CLI, les entraînements et les notebooks."""

import numpy as np


def binary_metrics(targets, predictions, available=None):
    targets, predictions = np.asarray(targets), np.asarray(predictions, dtype=bool)
    mask = (
        np.ones(len(targets), dtype=bool)
        if available is None
        else np.asarray(available, dtype=bool)
    )
    if targets.shape != predictions.shape or mask.shape != targets.shape:
        raise ValueError("Les cibles, prédictions et masques doivent être alignés.")
    if not np.isin(targets[mask], [0, 1]).all():
        raise ValueError("Les cibles supervisées doivent être binaires.")
    y, p = targets[mask].astype(bool), predictions[mask]
    tp = int(np.sum(y & p))
    fp = int(np.sum(~y & p))
    fn = int(np.sum(y & ~p))
    tn = int(np.sum(~y & ~p))
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        evaluated=int(mask.sum()),
        masked=int((~mask).sum()),
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / (tp + fn) if tp + fn else None,
        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    )


def choose_threshold(targets, probabilities, available=None):
    """Calibrage sur validation seulement ; les égalités favorisent le seuil élevé."""
    rows = [
        (float(t), binary_metrics(targets, np.asarray(probabilities) >= t, available))
        for t in np.linspace(0.05, 0.95, 19)
    ]
    threshold, score = max(
        rows,
        key=lambda item: (item[1]["f1"] if item[1]["f1"] is not None else -1, item[0]),
    )
    return dict(
        threshold=threshold,
        score=score,
        selection="validation_only",
        candidates=[dict(threshold=t, **m) for t, m in rows],
    )


def grouped_metrics(targets, predictions, groups, available=None):
    targets, predictions, groups = map(np.asarray, (targets, predictions, groups))
    mask = (
        np.ones(len(targets), dtype=bool)
        if available is None
        else np.asarray(available, dtype=bool)
    )
    rows = [
        dict(
            episode=int(group),
            **binary_metrics(
                targets[groups == group],
                predictions[groups == group],
                mask[groups == group],
            ),
        )
        for group in np.unique(groups)
    ]
    return dict(
        micro=binary_metrics(targets, predictions, mask),
        episodes=rows,
        macro={
            key: float(np.mean(values)) if values else None
            for key in ["precision", "recall", "f1"]
            if (values := [r[key] for r in rows if r[key] is not None]) or True
        },
        statistical_unit="episode; window counts are correlated descriptive measures",
    )
