# Detailed Ruff Analysis

Total issues: 356

## Summary by rule

- **E501**: 173
- **F405**: 73
- **I001**: 72
- **E402**: 16
- **F403**: 6
- **F722**: 4
- **F841**: 4
- **F401**: 3
- **F811**: 3
- **F821**: 1
- **E731**: 1

## Summary by file

- /workspace/experiments/functional/smolyak_scaling/render_smolyak_scaling_report.py: 81
- /workspace/python/jax_util/base/linearoperator.py: 42
- /workspace/python/jax_util/base/__init__.py: 30
- /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py: 28
- /workspace/experiments/functional/smolyak_hlo/run_smolyak_hlo_case.py: 24
- /workspace/python/jax_util/base/nonlinearoperator.py: 14
- /workspace/python/tests/experiment_runner/test_subprocess_scheduler_unit.py: 8
- /workspace/python/tests/solvers/test_solver_internal_branches.py: 8
- /workspace/python/jax_util/optimizers/pdipm.py: 7
- /workspace/python/jax_util/base/protocols.py: 6
- /workspace/python/jax_util/experiment_runner/subprocess_scheduler.py: 6
- /workspace/python/test.py: 5
- /workspace/python/tests/experiment_runner/test_subprocess_scheduler.py: 5
- /workspace/python/tests/functional/test_protocols_and_smolyak_helpers.py: 5
- /workspace/python/tests/neuralnetwork/test_layer_utils_and_training.py: 5
- /workspace/python/tests/solvers/test_slq.py: 5
- /workspace/python/jax_util/solvers/lobpcg.py: 4
- /workspace/python/tests/functional/test_smolyak.py: 4
- /workspace/python/jax_util/functional/smolyak.py: 3
- /workspace/python/jax_util/solvers/_minres.py: 3
- /workspace/python/jax_util/solvers/kkt_solver.py: 3
- /workspace/python/tests/experiment_runner/test_gpu_scheduler.py: 3
- /workspace/python/tests/experiment_runner/test_runner.py: 3
- /workspace/python/tests/experiment_runner/test_standard_scheduler.py: 3
- /workspace/python/tests/functional/test_integrate.py: 3
- /workspace/python/jax_util/solvers/slq.py: 2
- /workspace/python/tests/base/test_nonlinearoperator.py: 2
- /workspace/python/tests/experiment_runner/_gpu_child_probe.py: 2
- /workspace/python/tests/neuralnetwork/test_neuralnetwork_forward.py: 2
- /workspace/python/tests/neuralnetwork/test_neuralnetwork_train.py: 2
- /workspace/python/typings/optax/__init__.pyi: 2
- /workspace/python/jax_util/base/_env_value.py: 1
- /workspace/python/jax_util/experiment_runner/__init__.py: 1
- /workspace/python/jax_util/experiment_runner/gpu_runner.py: 1
- /workspace/python/jax_util/experiment_runner/protocols.py: 1
- /workspace/python/jax_util/experiment_runner/runner.py: 1
- /workspace/python/jax_util/functional/__init__.py: 1
- /workspace/python/jax_util/functional/monte_carlo.py: 1
- /workspace/python/jax_util/functional/protocols.py: 1
- /workspace/python/jax_util/neuralnetwork/__init__.py: 1
- /workspace/python/jax_util/neuralnetwork/layer_utils.py: 1
- /workspace/python/jax_util/neuralnetwork/neuralnetwork.py: 1
- /workspace/python/jax_util/neuralnetwork/protocols.py: 1
- /workspace/python/jax_util/neuralnetwork/sequential_train.py: 1
- /workspace/python/jax_util/neuralnetwork/train.py: 1
- /workspace/python/jax_util/optimizers/__init__.py: 1
- /workspace/python/jax_util/optimizers/protocols.py: 1
- /workspace/python/jax_util/solvers/__init__.py: 1
- /workspace/python/jax_util/solvers/_check_mv_operator.py: 1
- /workspace/python/jax_util/solvers/pcg.py: 1
- /workspace/python/tests/base/test_base_exports.py: 1
- /workspace/python/tests/base/test_env_value.py: 1
- /workspace/python/tests/base/test_env_value_helpers.py: 1
- /workspace/python/tests/base/test_linearoperator.py: 1
- /workspace/python/tests/base/test_linearoperator_branches.py: 1
- /workspace/python/tests/base/test_protocols.py: 1
- /workspace/python/tests/conftest.py: 1
- /workspace/python/tests/hlo/test_hlo_dump.py: 1
- /workspace/python/tests/hlo/test_hlo_dump_helpers.py: 1
- /workspace/python/tests/optimizers/test_pdipm.py: 1
- /workspace/python/tests/solvers/test_check_mv_operator.py: 1
- /workspace/python/tests/solvers/test_kkt_solver.py: 1
- /workspace/python/tests/solvers/test_lobpcg.py: 1
- /workspace/python/tests/solvers/test_matrix_util.py: 1
- /workspace/python/tests/solvers/test_minres.py: 1
- /workspace/python/tests/solvers/test_pcg.py: 1
- /workspace/scripts/hlo/summarize_hlo_jsonl.py: 1

## Top files (by issue count) with sample issues

### /workspace/experiments/functional/smolyak_scaling/render_smolyak_scaling_report.py — 81 issues
- E501: 80
  - L57: Line too long (103 > 100)
  - L138: Line too long (111 > 100)
  - L206: Line too long (120 > 100)
- I001: 1
  - L2: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/base/linearoperator.py — 42 issues
- F405: 36
  - L15: `Matrix` may be undefined, or defined from star imports
  - L15: `Matrix` may be undefined, or defined from star imports
  - L26: `Vector` may be undefined, or defined from star imports
- E501: 4
  - L74: Line too long (124 > 100)
  - L80: Line too long (115 > 100)
  - L102: Line too long (128 > 100)
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- F403: 1
  - L4: `from .protocols import *` used; unable to detect undefined names

### /workspace/python/jax_util/base/__init__.py — 30 issues
- F405: 25
  - L11: `DEFAULT_DTYPE` may be undefined, or defined from star imports
  - L12: `EPS` may be undefined, or defined from star imports
  - L13: `DEBUG` may be undefined, or defined from star imports
- F403: 4
  - L4: `from .protocols import *` used; unable to detect undefined names
  - L5: `from ._env_value import *` used; unable to detect undefined names
  - L6: `from .linearoperator import *` used; unable to detect undefined names
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py — 28 issues
- E501: 24
  - L234: Line too long (101 > 100)
  - L237: Line too long (106 > 100)
  - L319: Line too long (107 > 100)
- I001: 3
  - L2: Import block is un-sorted or un-formatted
  - L26: Import block is un-sorted or un-formatted
  - L320: Import block is un-sorted or un-formatted
- E402: 1
  - L26: Module level import not at top of file

### /workspace/experiments/functional/smolyak_hlo/run_smolyak_hlo_case.py — 24 issues
- E501: 16
  - L151: Line too long (148 > 100)
  - L153: Line too long (135 > 100)
  - L155: Line too long (167 > 100)
- E402: 6
  - L43: Module level import not at top of file
  - L44: Module level import not at top of file
  - L45: Module level import not at top of file
- I001: 2
  - L2: Import block is un-sorted or un-formatted
  - L43: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/base/nonlinearoperator.py — 14 issues
- F405: 12
  - L15: `Vector` may be undefined, or defined from star imports
  - L15: `Vector` may be undefined, or defined from star imports
  - L16: `Vector` may be undefined, or defined from star imports
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- F403: 1
  - L4: `from .protocols import *` used; unable to detect undefined names

### /workspace/python/tests/experiment_runner/test_subprocess_scheduler_unit.py — 8 issues
- E501: 6
  - L146: Line too long (107 > 100)
  - L150: Line too long (108 > 100)
  - L157: Line too long (127 > 100)
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- F401: 1
  - L5: `signal` imported but unused

### /workspace/python/tests/solvers/test_solver_internal_branches.py — 8 issues
- E501: 5
  - L36: Line too long (125 > 100)
  - L59: Line too long (109 > 100)
  - L65: Line too long (102 > 100)
- F401: 2
  - L7: `jax` imported but unused
  - L22: `jax_util.solvers.lobpcg.BlockEigenState` imported but unused
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/optimizers/pdipm.py — 7 issues
- E501: 6
  - L202: Line too long (103 > 100)
  - L210: Line too long (116 > 100)
  - L253: Line too long (101 > 100)
- I001: 1
  - L19: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/base/protocols.py — 6 issues
- F722: 4
  - L17: Syntax error in forward annotation: Expected an expression
  - L19: Syntax error in forward annotation: Unexpected token at the end of an expression
  - L21: Syntax error in forward annotation: Expected an expression
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- F821: 1
  - L18: Undefined name `n`

### /workspace/python/jax_util/experiment_runner/subprocess_scheduler.py — 6 issues
- E501: 5
  - L55: Line too long (112 > 100)
  - L108: Line too long (104 > 100)
  - L188: Line too long (105 > 100)
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/python/test.py — 5 issues
- I001: 3
  - L3: Import block is un-sorted or un-formatted
  - L6: Import block is un-sorted or un-formatted
  - L18: Import block is un-sorted or un-formatted
- E402: 1
  - L18: Module level import not at top of file
- E731: 1
  - L36: Do not assign a `lambda` expression, use a `def`

### /workspace/python/tests/experiment_runner/test_subprocess_scheduler.py — 5 issues
- E402: 2
  - L13: Module level import not at top of file
  - L15: Module level import not at top of file
- E501: 2
  - L147: Line too long (105 > 100)
  - L152: Line too long (111 > 100)
- I001: 1
  - L13: Import block is un-sorted or un-formatted

### /workspace/python/tests/functional/test_protocols_and_smolyak_helpers.py — 5 issues
- E501: 4
  - L99: Line too long (108 > 100)
  - L101: Line too long (105 > 100)
  - L109: Line too long (105 > 100)
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/python/tests/neuralnetwork/test_layer_utils_and_training.py — 5 issues
- E501: 5
  - L28: Line too long (114 > 100)
  - L47: Line too long (157 > 100)
  - L48: Line too long (177 > 100)

### /workspace/python/tests/solvers/test_slq.py — 5 issues
- F811: 3
  - L15: Redefinition of unused `DEFAULT_DTYPE` from line 12: `DEFAULT_DTYPE` redefined here
  - L16: Redefinition of unused `LinOp` from line 13: `LinOp` redefined here
  - L17: Redefinition of unused `Vector` from line 14: `Vector` redefined here
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- E501: 1
  - L139: Line too long (103 > 100)

### /workspace/python/jax_util/solvers/lobpcg.py — 4 issues
- F841: 2
  - L186: Local variable `theta` is assigned to but never used
  - L190: Local variable `Y_X` is assigned to but never used
- E501: 1
  - L6: Line too long (114 > 100)
- I001: 1
  - L15: Import block is un-sorted or un-formatted

### /workspace/python/tests/functional/test_smolyak.py — 4 issues
- E501: 3
  - L196: Line too long (105 > 100)
  - L197: Line too long (105 > 100)
  - L276: Line too long (105 > 100)
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/functional/smolyak.py — 3 issues
- E501: 2
  - L269: Line too long (111 > 100)
  - L291: Line too long (117 > 100)
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/solvers/_minres.py — 3 issues
- E501: 2
  - L122: Line too long (104 > 100)
  - L248: Line too long (102 > 100)
- I001: 1
  - L14: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/solvers/kkt_solver.py — 3 issues
- F841: 2
  - L179: Local variable `n_dual` is assigned to but never used
  - L191: Local variable `n_dual` is assigned to but never used
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/python/tests/experiment_runner/test_gpu_scheduler.py — 3 issues
- I001: 2
  - L1: Import block is un-sorted or un-formatted
  - L16: Import block is un-sorted or un-formatted
- E402: 1
  - L16: Module level import not at top of file

### /workspace/python/tests/experiment_runner/test_runner.py — 3 issues
- I001: 2
  - L1: Import block is un-sorted or un-formatted
  - L16: Import block is un-sorted or un-formatted
- E402: 1
  - L16: Module level import not at top of file

### /workspace/python/tests/experiment_runner/test_standard_scheduler.py — 3 issues
- I001: 2
  - L1: Import block is un-sorted or un-formatted
  - L12: Import block is un-sorted or un-formatted
- E402: 1
  - L12: Module level import not at top of file

### /workspace/python/tests/functional/test_integrate.py — 3 issues
- E501: 2
  - L27: Line too long (103 > 100)
  - L72: Line too long (108 > 100)
- I001: 1
  - L1: Import block is un-sorted or un-formatted

### /workspace/python/jax_util/solvers/slq.py — 2 issues
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- E501: 1
  - L186: Line too long (105 > 100)

### /workspace/python/tests/base/test_nonlinearoperator.py — 2 issues
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- E402: 1
  - L11: Module level import not at top of file

### /workspace/python/tests/experiment_runner/_gpu_child_probe.py — 2 issues
- E402: 1
  - L14: Module level import not at top of file
- E501: 1
  - L84: Line too long (105 > 100)

### /workspace/python/tests/neuralnetwork/test_neuralnetwork_forward.py — 2 issues
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- E501: 1
  - L16: Line too long (114 > 100)

### /workspace/python/tests/neuralnetwork/test_neuralnetwork_train.py — 2 issues
- I001: 1
  - L1: Import block is un-sorted or un-formatted
- E501: 1
  - L19: Line too long (114 > 100)


Full raw ruff JSON is available at `reports/static-analysis/ruff.json`.