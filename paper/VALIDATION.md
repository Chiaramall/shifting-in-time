# Export validation

Validated on 2026-10-08 with the existing Python 3.12 WarpRec environment:

- 16 existing unit tests passed.
- All four YAML files contain nine models, each with `max_seq_len: 20`.
- All nine models trained for one complete epoch on a synthetic 8-user,
  80-item dataset, using reduced hidden dimensions and CPU execution.
- All nine checkpoints were reloaded and evaluated under all 12 perturbation
  conditions (three positions × four intensities), with completed output markers.
- Python wheel built successfully without reinstalling dependencies.
- Shell runner passed Bash syntax validation.
- Export contains no model weights, raw datasets, logs, or historical Git objects.

The synthetic integration check validates pipeline connectivity; it does not
reproduce the paper’s numerical results. A clean-environment dependency install
and full training on the original prepared datasets were not performed.

Qualitative artifacts added on 2026-10-08: the two archived examples were
checked against their JSON records. All six FRBO scores, the length-preserving
sequence reconstructions, and the genre/artist counts match. The original full
training histories were not revalidated. Only textual examples and evidence are
included; no qualitative generation scripts were added.

Aggregate result tables were removed from the export because their values do not
consistently match the supplied manuscript. The qualitative examples are retained.

The qualitative LaTeX tables, PDF preview, and table-specific display notes were
removed at the user’s request.
