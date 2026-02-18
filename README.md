# Settlers of Catan AI

## 1. Core Strategic Principles (Encoded as Heuristics)
A competent heuristic agent in Catan should prioritize:
- Expected resource production (pip count maximization)
- Settlement expansion > city upgrade > road spam
- Access to diverse resources
- Blocking opponents at high-probability intersections
- Efficient resource spending (avoid hoarding)
- Victory point acceleration once ≥ 7 VP
- Robber targeting of leader
- These principles can be translated into a scoring function.
