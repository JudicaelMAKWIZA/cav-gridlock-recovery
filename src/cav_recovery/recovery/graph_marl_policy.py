"""Choix neuronal local d'une option au sein du protocole déterministe P4."""

import torch

from .cooperative import CooperativePolicy, option_bank
from .graph_policy import local_graph
from .verified_passage_policy import GuardEnvelopePassagePolicy


class P4GraphPolicy(GuardEnvelopePassagePolicy):
    """P4 gère les accords et les passages ; le réseau choisit le dégagement.

    Le masque provient des options locales recevables de REC-COOP-2. Leurs
    scores et leur ordre ne sont jamais transmis à l'acteur. Le garde vérifie
    encore l'action finalement proposée avant son exécution native.
    """

    def __init__(self, ego, endpoint, contract, model, record, *, sample=True):
        super().__init__(ego, endpoint, contract)
        self.model = model
        self.record = record
        self.sample = sample

    def candidates(self, view, guard):
        bank = option_bank()
        admissible = CooperativePolicy.candidates(self, view, guard)
        mask = [index == 0 or action in admissible for index, action in enumerate(bank)]
        if not any(mask):
            raise ValueError("Aucune option locale disponible, y compris l'attente.")
        nodes, adjacency = local_graph(view)
        with torch.no_grad():
            logits, value = self.model(
                torch.from_numpy(nodes), torch.from_numpy(adjacency)
            )
            permitted = logits.masked_fill(~torch.tensor(mask), -1e9)
            distribution = torch.distributions.Categorical(logits=permitted)
            index = int(distribution.sample() if self.sample else permitted.argmax())
            logprob = float(distribution.log_prob(torch.tensor(index)))
        self.record(dict(
            ego=view.ego, at_s=view.at_s, phase="physical_option",
            features=nodes.tolist(), adjacency=adjacency.tolist(), mask=mask,
            action=index, old_logprob=logprob, value=float(value),
            probability=float(distribution.probs[index]),
            admissible_count=sum(mask),
            chosen=dict(speed_mps=bank[index].speed_mps,
                        steering=bank[index].steering, reason=bank[index].reason),
        ))
        return [] if index == 0 else [bank[index]]
