
# ===== Trust Calibration and Consensus =====
def trust_calibration(agents, outcomes):
    for agent, outcome in zip(agents, outcomes):
        if "Blocked" in outcome or "harm" in outcome or "misinfo" in outcome.lower():
            agent.trust *= 0.7
        elif "Approved" in outcome or "safety" in outcome:
            agent.trust *= 1.05
        else:
            agent.trust *= 0.98
        agent.trust = max(0.05, min(agent.trust, 1.5))
    return agents

def weighted_consensus(agents, proposals):
    scores = {}
    for agent, proposal in zip(agents, proposals):
        if agent.trust > 0.15:
            scores[proposal] = scores.get(proposal, 0) + agent.trust
    if not scores:
        return "All proposals blocked by ethical constraints or low trust."
    top_score = max(scores.values())
    winners = [p for p, s in scores.items() if s == top_score]
    return choice(winners)
