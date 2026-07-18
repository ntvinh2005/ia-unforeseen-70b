# IA Unforeseen Behavior Audit

**→ [Documentation](docs/) — All guides, workflows, troubleshooting**

This workspace now has two deliberately separate subsystems:

- the existing Meta-IA training and sanity-evaluation scripts; and
- a staged, file-based unforeseen-behavior audit under `src/audit` and
  `src/meta_ia_eval`.

The audit pipeline follows one hard rule: prompt generation and every judge run
use a clean base model in a separate process. A production process therefore
loads only one of `BASE`, `TARGET`, `JUDGE`, `PROMPT_GEN`, `BASE_IA`, or
`TARGET_IA`, then communicates with the next stage only through validated
JSON/JSONL artifacts.

## Setup

The numbered scripts can run directly from the repository; installing the
package is optional. In the HiPerGator environment, the expected model runtime
dependencies are PyTorch, Transformers, PEFT, Accelerate, and
`huggingface-hub`.

```bash
cp configs/unforeseen_audit.example.yaml configs/unforeseen_audit.yaml
cp configs/acquisition_prompts.example.jsonl configs/acquisition_prompts.jsonl
# Replace every REPLACE_* value and create the held-out acquisition prompt bank.
python -m pip install -e '.[runtime,config]'

export PROJECT="${PROJECT:-$PWD}"
export AUDIT_CONFIG="$PROJECT/configs/unforeseen_audit.yaml"
export AUDIT_OUTPUT="$PROJECT/outputs/unforeseen_audit_v1"
```

The example `.yaml` is also valid JSON, so it remains loadable without PyYAML.
All adapter paths/checkpoints, decoding values, thresholds, and prompt-bank
versions should be finalized before looking at audit results.

The top-level `profile` is the canonical prompt-budget selector. `mvp` creates
40 single-turn plus 8 multi-turn discovery prompts, then 20 development and 24
test prompts per accepted hypothesis. `full` creates 120 plus 15 discovery
prompts, then 20 development and 44 test prompts per hypothesis. The matching
count fields under `discovery` and `verification` are assertions that catch
configuration drift; they are not independent custom-budget controls.

## Staged workflow

Every command takes `--config configs/unforeseen_audit.yaml`. Model-producing
stages expose conditions explicitly so BASE and TARGET can be separate SLURM
jobs.

```bash
# 1. Confirm that the behavior adapter learned its intended narrow behavior.
python scripts/01_verify_adapter_acquisition.py --config "$AUDIT_CONFIG" --phase generate --condition BASE
python scripts/01_verify_adapter_acquisition.py --config "$AUDIT_CONFIG" --phase generate --condition TARGET
python scripts/01_verify_adapter_acquisition.py --config "$AUDIT_CONFIG" --phase grade
python scripts/01_verify_adapter_acquisition.py --config "$AUDIT_CONFIG" --phase summarize

# 2–5. Blind discovery and hypothesis triage.
python scripts/02_generate_discovery_prompts.py --config "$AUDIT_CONFIG"
python scripts/03_generate_discovery_rollouts.py --config "$AUDIT_CONFIG" --condition BASE
python scripts/03_generate_discovery_rollouts.py --config "$AUDIT_CONFIG" --condition TARGET
python scripts/04_run_open_diff_judge.py --config "$AUDIT_CONFIG"
python scripts/05_cluster_hypotheses.py --config "$AUDIT_CONFIG"

# Human-review clustered_candidates.json. Use the strict example as a template.
cp configs/hypotheses_human_reviewed.example.json \
  "$AUDIT_OUTPUT/hypotheses/human_reviewed.json"
# Replace placeholders and retain only hypotheses intentionally triaged.
python scripts/06_generate_targeted_evals.py --config "$AUDIT_CONFIG"

# 7–8a. Development split: tune and lock the behavior-grader prompt here.
python scripts/07_generate_verification_rollouts.py --config "$AUDIT_CONFIG" --split dev --condition BASE
python scripts/07_generate_verification_rollouts.py --config "$AUDIT_CONFIG" --split dev --condition TARGET
python scripts/08_grade_verification.py --config "$AUDIT_CONFIG" --split dev --phase grade
python scripts/08_grade_verification.py --config "$AUDIT_CONFIG" --split dev --phase summarize

# 7–8b. Held-out test split; do not revise the hypothesis or grader afterward.
python scripts/07_generate_verification_rollouts.py --config "$AUDIT_CONFIG" --split test --condition BASE
python scripts/07_generate_verification_rollouts.py --config "$AUDIT_CONFIG" --split test --condition TARGET
python scripts/08_grade_verification.py --config "$AUDIT_CONFIG" --split test --phase grade
python scripts/08_grade_verification.py --config "$AUDIT_CONFIG" --split test --phase summarize

# 9. Freeze human-approved labels before any Meta-IA output is generated.
cp configs/human_label_reviews.example.json \
  "$AUDIT_OUTPUT/verification/human_label_reviews.json"
cp configs/calibration.example.jsonl \
  "$AUDIT_OUTPUT/verification/calibration.jsonl"
# Replace every placeholder with IDs from accepted hypotheses, TARGET test
# rollouts, and test judgments. Human scores use the same 0..3 rubric.
python scripts/09_finalize_verified_labels.py --config "$AUDIT_CONFIG"

# 10. Generate each introspection condition in its own job, then grade cleanly.
python scripts/10_evaluate_meta_ia.py --config "$AUDIT_CONFIG" --phase rollouts --condition TARGET
python scripts/10_evaluate_meta_ia.py --config "$AUDIT_CONFIG" --phase rollouts --condition BASE_IA
python scripts/10_evaluate_meta_ia.py --config "$AUDIT_CONFIG" --phase rollouts --condition TARGET_IA
python scripts/10_evaluate_meta_ia.py --config "$AUDIT_CONFIG" --phase grade
python scripts/10_evaluate_meta_ia.py --config "$AUDIT_CONFIG" --phase summarize
```

Use `--help` on a stage for its artifact overrides and safe rerun controls. The
generic SLURM launcher accepts the stage number followed by the same stage
arguments. Every launcher writes to `logs/`; create that directory before
`sbatch`, because Slurm opens output files before the script body can run:

```bash
mkdir -p "$PROJECT/logs" # Slurm opens the log path before the job script starts.
sbatch slurm/run_unforeseen_audit_stage.slurm 03 --condition BASE
sbatch slurm/run_unforeseen_audit_stage.slurm 03 --condition TARGET
```

The manual inputs are strict, versioned artifacts:

- `hypotheses/human_reviewed.json` is a `schema_version: 1` wrapper with a
  `hypotheses` array. Accepted entries use
  `status: accepted_for_verification` and a non-null `classification`.
- `verification/human_label_reviews.json` is a `schema_version: 1` wrapper with
  a `reviews` array. Each review records approval, the training relationship,
  and clear TARGET test-rollout IDs.
- `verification/calibration.jsonl` has exactly `hypothesis_id`, `rollout_id`,
  and integer `human_score` (0 through 3) on every line. IDs must reference
  Stage-08 test grades for that hypothesis. The four example rows illustrate
  the rubric only; expand this to the preregistered 40 human-scored examples
  before Stage 9.

Examples live in `configs/`; placeholders are deliberately invalid experiment
identities and must be replaced rather than treated as data.

## Artifact contract

The canonical experiment tree is:

```text
outputs/unforeseen_audit_v1/
├── config.yaml
├── provenance.json
├── frozen_manifest.json
├── acquisition/
├── prompts/
│   ├── discovery.jsonl
│   ├── targeted_dev.jsonl
│   └── targeted_test.jsonl
├── rollouts/{base,target}/
├── discovery_judgments/
├── hypotheses/
│   ├── raw_candidates.jsonl
│   ├── clustered_candidates.json
│   └── human_reviewed.json
├── verification/
│   ├── {dev,test}_judgments.jsonl
│   ├── {dev,test}_metrics.json
│   ├── {dev,test}_bootstrap_results.json
│   ├── human_label_reviews.json
│   └── calibration.jsonl
├── verified_labels/
│   ├── labels_v1.jsonl
│   └── labels_v1.jsonl.manifest.json
└── meta_ia_evaluation/
    ├── rollouts_{target,base_ia,target_ia}.jsonl
    ├── rollouts.jsonl
    ├── judgments.jsonl
    └── metrics.json
```

Discovery prompts cannot contain a known behavior label. Every rollout stores
the named condition plus `adapter_active` and `meta_ia_active`, and schema
validation rejects contradictory compositions. Frozen labels are create-once;
they cannot be rewritten after Meta-IA evaluation begins.

## Legacy Meta-IA training

The operational multi-adapter launcher remains:

```bash
python scripts/build_auditbench_llama70b_manifest.py \
  --output datasets/manifests/auditbench_llama70b_manifest.json
python scripts/train_meta_ia_multi_adapter.py \
  --config scripts/configs/train_meta_ia_multi_adapter.json
```

Remote adapter and dataset revisions must be full 40-character commit SHAs.
The generated schema-v2 manifest also records and verifies the SHA-256 and byte
size of every `eval.jsonl` used by training or evaluation. The nested
`repos/introspection-adapters` checkout contains a user-modified dynamic LRU
implementation in `src/finetuning/metalora.py`; this refactor intentionally
does not overwrite that work.

## Verification

Pure pipeline logic is covered without loading model weights:

```bash
python -m pytest tests -q
python -m compileall -q src scripts
```

Full 70B generation is expected to run on the GPU cluster and is validated by
the persisted provenance, composition flags, cache keys, and stage artifacts.
