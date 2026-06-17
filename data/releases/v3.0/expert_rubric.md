# Expert Reference-Decomposition Rubric

Experts were asked to construct architecture-level reference decompositions independently before consensus review. The original package structure was available as contextual information but was not treated as a fixed answer.

## Criteria

1. **Business-logic cohesion**: entities assigned to the same module should implement related responsibilities.
2. **Inter-module dependency minimization**: avoid unnecessary cross-module calls, inheritance links, and shared mutable dependencies.
3. **Naming consistency**: package and class names should be semantically coherent within a module.
4. **Service-boundary feasibility**: candidate modules should be plausible units for later architectural review or microservice-boundary analysis.
5. **Manual-review flagging**: highly coupled or ambiguous entities should be flagged rather than forced into unsupported migrations.

## Consensus Protocol

1. Each expert produced an independent decomposition.
2. Pairwise disagreements were logged by entity and module assignment.
3. Consensus meetings resolved disagreements using the four criteria above.
4. Cohen's kappa was computed from the independent assignments before consensus.
5. The final consensus decomposition was used as the reference for MoJoFM-style comparison.
