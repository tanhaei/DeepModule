# DeepModule Output Schema

## `modularization_recommendations.csv`

- `Class`: fully qualified class or file-level entity name.
- `Recommended_Module`: integer module assignment produced by DeepModule.

## `preprocessing_log.csv`

- `file`: relative file path inspected by the preprocessing stage.
- `action`: `included` or `excluded`.
- `reason`: rule explaining the action, such as generated file, test fixture, vendor library, malformed parse tree, isolated node, or short file.

## `boundary_report.csv`

- `source`: source entity of the inter-module dependency.
- `target`: target entity of the inter-module dependency.
- `source_module`: predicted source module.
- `target_module`: predicted target module.
- `dependency_type`: extracted static dependency type when available.
- `attention_weight`: GAT attention weight when available.
- `semantic_similarity`: cosine similarity between source and target representations.
- `recommendation`: suggested review action, such as move class, split package, extract facade, or cross-service API candidate.

## `embeddings.npy`

NumPy array containing learned node embeddings in the same order as the internal class-name list.

## `test_skeletons/`

Generated JUnit-style interface-check skeletons for boundary review. These files are review artifacts and do not prove full behavioral equivalence.
