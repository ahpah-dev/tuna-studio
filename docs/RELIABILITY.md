# Workbench and reliability update

Verified locally on 23 October 2026.

The checked flow is dataset validation and run configuration in the browser →
local API → saved metadata → training/export worker lifecycle, plus chat
submission → retained reply → restored browser conversation.

## Error recovery

- Browser settings and chat history validate stored types before use. Invalid
  values fall back to usable defaults; valid conversation pairs remain readable.
- Interrupted chat submissions restore the draft rather than keeping the
  composer disabled. Completed replies can be recovered by request ID without
  replacing the current worker status. Stop remains available for an active
  reply even when the browser lost its pending request ID.
- Storage write failures produce a visible warning. Connection errors clear
  after a successful poll while actionable operation errors stay visible.
- API errors distinguish connection failure, timeout, unreadable responses, and
  server validation. Request timeouts prevent indefinitely disabled controls.
- Independent refresh failures preserve the last valid runs/datasets. Connection
  status updates while typing or while a dialog is open.
- Dataset import and run configuration guard duplicate submissions. Numeric
  settings are validated before requesting training. Late requests and file
  reads cannot overwrite a different dialog or reopen a closed dialog.
- Malformed settings, incomplete dataset metadata, and damaged optional run
  metadata no longer take down startup or listing. Run IDs must match their
  directory. Invalid entries are skipped rather than deleted.
- Windows training and export cancellation targets the interpreter process tree,
  including the child of a virtual-environment launcher, with bounded waits and
  escalation. This prevents an orphan child from retaining GPU memory.

## Interface

The layout uses a graphite navigation rail, neutral work surfaces, a restrained
teal accent, local system fonts, sharper controls, and a continuous metric strip.
Recent runs and runtime status precede the method picker. The workflow welcome
appears only in an empty workspace. Chat, dataset import, and run/export dialogs
use the same visual language. Reduced-motion preferences are respected.

## Validation

- All **92 Python tests passed**, including seven new persistence/cancellation
  regressions and the existing training, model import/export, and chat checks.
- Dashboard and chat Node regression checks passed: malformed storage, interrupted
  submissions, server session changes, API failures, partial refresh, duplicate
  actions, numeric guards, and stale dialog/file responses.
- Both JavaScript entry points passed syntax checks; Git diff checks passed.
- In the live browser, malformed JSONL returned its line-specific validation
  error without adding a dataset. A zero learning rate was rejected inline
  without creating a run. The existing seven runs and one dataset remained.
- Stopping the idle server showed an automatic reconnect banner while preserving
  displayed data. Restarting restored the connected state and existing chat
  conversation. CUDA readiness was detected on the RTX 5070.
- The 390 px chat layout had no horizontal document overflow. Desktop overview,
  runs, import, setup, and chat surfaces were inspected.

This is regression and workflow verification, not a measured percentage reduction
in every possible error. Model compatibility, available memory, dataset quality,
and model-generated reasoning behavior retain their documented requirements.
