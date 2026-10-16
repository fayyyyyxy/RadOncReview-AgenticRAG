# Local data

Place RadOnc-Scenario-Bank-v2-Combined-Excel.xlsx in this directory.

All contents except this README are ignored by Git. Obtain the workbook from
the team's approved source; do not commit patient-case inputs or exports.

The loader writes processed/development_examples.jsonl and
processed/baseline_inputs.jsonl here. These are development-only exports;
locked-test rows remain in the source workbook. The baseline input file has
only case_id, vignette, and clinical_question, with no attending plans.

Rerunning the loader replaces generated files only after validation succeeds.
