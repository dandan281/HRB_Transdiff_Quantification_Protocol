# Protocol approval

On 2026-07-16, the biological reviewer:

- reviewed and approved `../ANNOTATION_PROTOCOL.md`, subject to making the four status definitions
  explicit;
- reviewed and approved the instructions and planned workflow in the `HUMAN` folder;
- did not claim that the dual-annotation pilot had been completed or adjudicated.

The approved status vocabulary is:

- `complete`: full independently traceable and measurable object;
- `border_truncated`: object continues beyond the captured image boundary;
- `occluded`: identity remains clear, but a material part is hidden;
- `ambiguous`: identity, connectivity, or boundary assignment cannot be resolved from the pixels.

On 2026-07-21, the same user clarified that no second annotator or adjudicator is available and
requested a feasible replacement plan. `../DEVELOPMENT_PLAN.md` now governs the project. The CL04
two-reviewer signature requirement and old dual-annotation gate are historical rather than active
blockers. The project will report single-operator test-retest evidence and will not claim inter-rater
agreement or consensus ground truth.
