"""Benchmark Suite

## Overview

This directory contains performance benchmarks for the jax_util library.
Unlike `python/tests/`, which verify correctness, benchmarks measure
_performance_ under specific conditions.

## Directories

- `functional/`: Benchmarks for integration methods (Smolyak, Monte Carlo, etc.)

## Quick Start

```bash
# Run all benchmarks and display results
python3 python/benchmark/functional/benchmark_smolyak_integrator.py

# Save results to JSON
python3 python/benchmark/functional/benchmark_smolyak_integrator.py /path/to/result.json
```

## Design Philosophy

- **Reproducible**: Run the same benchmark on different machines, compare results
- **Fast**: All benchmarks complete within minutes (target: 1-2 minutes total)
- **Quantitative**: Measure init time, integral time, scaling factors
- **Decoupled**: Benchmarks run independently, no shared state

## Typical Workflow

1. Implement a change to `jax_util.functional.smolyak`
2. Run benchmark before and after change
3. Compare JSON output to quantify improvement/regression
4. Document significant changes in `notes/knowledge/`

## Reference

- Policy: [documents/conventions/python/20_benchmark_policy.md](../../../documents/conventions/python/20_benchmark_policy.md)
- Benchmark vs Experiment: [notes/knowledge/benchmark_vs_experiment.md](../../../notes/knowledge/benchmark_vs_experiment.md)
"""
