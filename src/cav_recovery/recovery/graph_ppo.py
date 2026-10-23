"""Retours d'options et mises à jour PPO à partir de choix locaux réels."""

from collections import defaultdict
import math

import torch


def _step_index(at_s: float, step_s: float, count: int) -> int:
    index = round(at_s / step_s)
    if abs(index * step_s - at_s) > 1e-5 or not 0 <= index <= count:
        raise ValueError(f"Décision hors de la grille physique : {at_s}")
    return index


def option_transitions(decisions, rewards, *, gamma, gae_lambda=0.95,
                       step_s=0.2, terminal=False, termination_times=None,
                       bootstrap_values=None):
    """Attribue à chaque choix son intervalle physique jusqu'au choix suivant.

    Le même signal d'équipe est partagé entre choix simultanés. Une arrivée
    native termine l'option de cet agent. À un horizon censuré, le dernier
    graphe local fournit la valeur de continuation ; sans cette mesure,
    l'option inachevée est écartée.
    """
    if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Actualisation ou lambda invalide.")
    if not all(math.isfinite(float(value)) for value in rewards):
        raise ValueError("Récompense non finie.")
    groups = defaultdict(list)
    for row in decisions:
        if len(row["mask"]) == 0 or not row["mask"][row["action"]]:
            raise ValueError("Choix neuronal hors masque.")
        if not math.isfinite(float(row["value"])):
            raise ValueError("Valeur critique non finie.")
        groups[row["ego"]].append(row)
    transitions = []
    termination_times = termination_times or {}
    bootstrap_values = bootstrap_values or {}
    for ego, sequence in groups.items():
        sequence = sorted(sequence, key=lambda row: row["at_s"])
        later_advantage = 0.0
        for offset in range(len(sequence) - 1, -1, -1):
            row = sequence[offset]
            start = _step_index(row["at_s"], step_s, len(rewards))
            successor = sequence[offset + 1] if offset + 1 < len(sequence) else None
            reached = ego in termination_times
            end = (_step_index(successor["at_s"], step_s, len(rewards))
                   if successor else
                   _step_index(termination_times[ego], step_s, len(rewards))
                   if reached else len(rewards))
            if end <= start:
                raise ValueError(f"Durée d'option non positive : {ego}")
            if (successor is None and not reached and not terminal
                    and ego not in bootstrap_values):
                later_advantage = 0.0
                continue
            duration = end - start
            interval_return = sum(
                gamma ** index * float(rewards[start + index])
                for index in range(duration)
            )
            continuation = gamma ** duration
            next_value = (float(successor["value"]) if successor else
                          float(bootstrap_values[ego]) if ego in bootstrap_values
                          and not reached else 0.0)
            delta = interval_return + continuation * next_value - float(row["value"])
            advantage = delta + (continuation * gae_lambda ** duration
                                 * later_advantage
                                 if successor else 0.0)
            later_advantage = advantage
            if sum(bool(allowed) for allowed in row["mask"]) <= 1:
                continue
            transitions.append(dict(
                **row, option_steps=duration, option_reward=interval_return,
                target_return=float(row["value"]) + advantage,
                advantage=advantage, terminal=successor is None and
                (reached or terminal), censored=successor is None and
                not reached and not terminal,
            ))
    return sorted(transitions, key=lambda row: (row["at_s"], row["ego"]))


def ppo_update_batch(model, optimizer, transitions, *, epochs, minibatch_size,
                     clip_ratio, value_coefficient, entropy_coefficient,
                     gradient_norm):
    """Une collecte on-policy, plusieurs époques et des minibatchs réels."""
    if not transitions:
        return dict(updates=0, decisions=0, reason="aucun_choix_libre")
    if minibatch_size < 1 or epochs < 1:
        raise ValueError("Taille de lot ou nombre d'époques invalide.")
    advantages = torch.tensor([row["advantage"] for row in transitions],
                              dtype=torch.float32)
    if not torch.isfinite(advantages).all():
        raise ValueError("Avantage non fini.")
    if len(advantages) > 1 and advantages.std(unbiased=False) > 1e-6:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False)
    with torch.no_grad():
        for row in transitions:
            logits, _ = model(torch.tensor(row["features"], dtype=torch.float32),
                              torch.tensor(row["adjacency"], dtype=torch.float32))
            mask = torch.tensor(row["mask"], dtype=torch.bool)
            distribution = torch.distributions.Categorical(
                logits=logits.masked_fill(~mask, -1e9))
            current = float(distribution.log_prob(torch.tensor(row["action"])))
            if abs(current - float(row["old_logprob"])) > 1e-4:
                raise ValueError("Le lot ne provient pas des poids courants.")
    losses = []
    for _ in range(epochs):
        for indices in torch.randperm(len(transitions)).split(minibatch_size):
            policy_terms, value_terms, entropies = [], [], []
            for index in indices.tolist():
                row = transitions[index]
                logits, value = model(
                    torch.tensor(row["features"], dtype=torch.float32),
                    torch.tensor(row["adjacency"], dtype=torch.float32))
                mask = torch.tensor(row["mask"], dtype=torch.bool)
                distribution = torch.distributions.Categorical(
                    logits=logits.masked_fill(~mask, -1e9))
                log_probability = distribution.log_prob(torch.tensor(row["action"]))
                ratio = torch.exp(log_probability - float(row["old_logprob"]))
                clipped = ratio.clamp(1 - clip_ratio, 1 + clip_ratio)
                policy_terms.append(-torch.minimum(
                    ratio * advantages[index], clipped * advantages[index]))
                value_terms.append((value - float(row["target_return"])) ** 2)
                entropies.append(distribution.entropy())
            policy_loss = torch.stack(policy_terms).mean()
            value_loss = torch.stack(value_terms).mean()
            entropy = torch.stack(entropies).mean()
            loss = (policy_loss + value_coefficient * value_loss
                    - entropy_coefficient * entropy)
            if not torch.isfinite(loss):
                raise ValueError("Perte PPO non finie.")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_norm)
            optimizer.step()
            losses.append(dict(total=float(loss.detach()),
                               policy=float(policy_loss.detach()),
                               value=float(value_loss.detach()),
                               entropy=float(entropy.detach()),
                               gradient_norm_before_clip=float(norm.detach())))
    return dict(updates=len(losses), decisions=len(transitions), losses=losses,
                option_steps_min=min(row["option_steps"] for row in transitions),
                option_steps_max=max(row["option_steps"] for row in transitions))
