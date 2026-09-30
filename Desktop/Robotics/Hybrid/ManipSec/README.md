# ManipSec

## ManipSec-Bench

A manipulation security benchmark for agentic policies.

### Environment Design

- **Task Suite:** 28 tasks grouped by physical workflow, with independent security annotations.
- **Agent Interface:** policy inputs, action outputs, allowed APIs, and prohibited access.

### Attack Design

- **Threat Model:** attacker access, objective, and budget.
- **Prompt Attack:** adversarial task instructions.
- **Scene Attack:** adversarial visual changes before execution.
- **Environment Attack:** adversarial disturbances during execution.

### Evaluation Protocol

- **Sandboxed Execution:** fixed API, time, and compute limits.
- **Metrics:** TSR, SSR, and ASR.

## ManipSec-ART

Evaluates frontier coding agents on ManipSec-Bench.

### Policy Generation

Coding agents generate executable manipulation policies.

### Clean Evaluation

Generated policies run on unmodified tasks.

### Adversarial Evaluation

The same policies run under prompt, scene, and environment attacks.

### Security Analysis

Results compare task competence, robustness, and compromise modes.

## ManipSec-Gen

Converts verified attack failures into security training data.

### Failure Collection

Collect successful attacks and failed policies.

### Secure Correction

Generate corrected policies and responses.

### Data Verification

Retain corrections that complete the task without violating the security objective.

### Model Training

Use verified examples for security fine-tuning.
