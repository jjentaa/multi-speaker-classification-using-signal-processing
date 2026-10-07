"""2.1.10 Clustering: decide *who* spoke each segment.

Online (real-time): every closed segment is compared with each known speaker
model using dBIC; it joins the closest one if dBIC < 0, otherwise it becomes a
new speaker (up to `max_speakers`). Two speaker models that become similar
(dBIC < 0) are merged, so an early wrong split does not stay forever.

Offline (optional refinement at the end of a recording): classic bottom-up
agglomerative BIC clustering over all segments.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .gauss import SpeakerStats, speaker_distance


class OnlineSpeakerClustering:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.speakers: dict[int, SpeakerStats] = {}   # speaker id -> model
        self._next_id = 0

    def assign(self, seg: SpeakerStats, speech_s: float) -> tuple[int, float, dict[int, float]]:
        """Returns (speaker id, confidence 0..1, dBIC to every speaker)."""
        cfg = self.cfg
        dists = {k: speaker_distance(seg, s, cfg.clu_lambda, cfg.pitch_weight) for k, s in self.speakers.items()}
        # full-covariance statistics of a very short segment are unreliable, so
        # only a long enough segment may open a new speaker
        can_create = len(self.speakers) < cfg.max_speakers and speech_s >= cfg.clu_min_new_s
        if not self.speakers:
            if speech_s < cfg.clu_min_new_s * 0.5:
                return -1, 0.0, dists
            return self._new(seg), 1.0, dists
        best = min(dists, key=dists.get)
        if dists[best] > 0 and can_create:
            return self._new(seg), _confidence(dists[best], seg.n), dists
        self.speakers[best].merge_(seg)
        self.speakers[best].decay_to(cfg.clu_max_frames)
        others = sorted(v for k, v in dists.items() if k != best)
        margin = (others[0] if others else 0.0) - dists[best]
        self._merge_similar(best)
        return best, _confidence(margin, seg.n), dists

    def _new(self, seg: SpeakerStats) -> int:
        s = SpeakerStats(seg.mfcc.d)
        s.merge_(seg)
        k = self._next_id
        self._next_id += 1
        self.speakers[k] = s
        return k

    def _merge_similar(self, k: int):
        """If the updated model k now looks like another speaker (dBIC < 0),
        the two were the same person all along: merge into the older id."""
        cfg = self.cfg
        for j in list(self.speakers):
            if j == k or j not in self.speakers or k not in self.speakers:
                continue
            if speaker_distance(self.speakers[k], self.speakers[j], cfg.clu_lambda, cfg.pitch_weight) < 0:
                keep, drop = min(j, k), max(j, k)
                self.speakers[keep].merge_(self.speakers.pop(drop))
                self.speakers[keep].decay_to(cfg.clu_max_frames)
                k = keep


def _confidence(margin: float, n: float) -> float:
    """Map a dBIC margin (per 100 frames) to 0..1 with a logistic curve."""
    z = margin / max(n, 1.0) * 100.0 / 20.0
    return float(1.0 / (1.0 + np.exp(-z)))


def agglomerative_bic(stats: list[SpeakerStats], cfg: Config) -> np.ndarray:
    """Bottom-up clustering: merge the closest pair while dBIC < 0
    (or while there are more than `max_speakers` clusters)."""
    clusters = {i: s for i, s in enumerate(stats)}
    members = {i: [i] for i in range(len(stats))}
    D: dict[tuple[int, int], float] = {}
    keys = list(clusters)
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            D[(keys[a], keys[b])] = speaker_distance(clusters[keys[a]], clusters[keys[b]], cfg.clu_lambda, cfg.pitch_weight)
    while len(clusters) > 1:
        (i, j), d = min(D.items(), key=lambda kv: kv[1])
        if d >= 0 and len(clusters) <= cfg.max_speakers:
            break
        clusters[i] = clusters[i].merged(clusters.pop(j))
        members[i] += members.pop(j)
        D = {k: v for k, v in D.items() if i not in k and j not in k}
        for k in clusters:
            if k != i:
                D[(min(i, k), max(i, k))] = speaker_distance(clusters[i], clusters[k], cfg.clu_lambda, cfg.pitch_weight)
    labels = np.zeros(len(stats), dtype=int)
    # number clusters in order of first appearance
    order = sorted(members, key=lambda k: min(members[k]))
    for new, k in enumerate(order):
        labels[members[k]] = new
    return labels
