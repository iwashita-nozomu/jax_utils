# Research Perspective Review

## Purpose

研究系の変更や実験証拠を、複数の独立視点で並列レビューします。

## Use This For

- benchmark protocol、artifact policy、reporting policy の大きな変更
- `experiments/`、`experiments/report/`、`notes/experiments/` を含む repo-wide review
- 研究 workflow、研究文書、比較設計の大きな整理
- method 採否や報告方針を main へ持ち帰る前の独立レビュー

## Must Read Before Reviewing

- `documents/research-workflow.md`
- `documents/experiment-workflow.md`
- `documents/experiment-critical-review.md`
- `agents/skills/project-review.md`
- `agents/skills/comprehensive-review.md`

## Inputs

- review scope
- 実験コード、report、notes、比較表、summary
- 必要なら benchmark protocol と artifact layout
- 必要なら前回 review の findings

## Outputs

- findings-first の perspective review
- perspective ごとの findings と follow-up
- `fix now`、`follow-up`、`delete-ok` の切り分け

## Mandatory Perspectives

1. `reproducibility`
   - provenance、seed、command、environment、rerunability
2. `scientific-computing`
   - incremental change、testing、automation、prototype discipline
3. `benchmark`
   - fairness、confounders、case mix、anti-pattern
4. `artifact`
   - code、script、raw result、environment、rerun package の十分性
5. `fair-data`
   - metadata、命名、再利用性、result path と note path の可搬性
6. `ml-science-reporting`
   - assumptions、limitations、uncertainty、checklist closure、reader-facing reporting

## Perspective To Subagent Mapping

- `reproducibility` -> `reproducibility_reviewer`
- `scientific-computing` -> `scientific_computing_reviewer`
- `benchmark` -> `benchmark_reviewer`
- `artifact` -> `artifact_reviewer`
- `fair-data` -> `fair_data_reviewer`
- `ml-science-reporting` -> `ml_science_reviewer`

## Canonical External References

- Sandve et al., Ten Simple Rules for Reproducible Computational Research
  - https://doi.org/10.1371/journal.pcbi.1003285
- Wilson et al., Best Practices for Scientific Computing
  - https://doi.org/10.1371/journal.pbio.1001745
- van der Kouwe et al., Benchmarking Crimes
  - https://doi.org/10.48550/arXiv.1801.02381
- ACM Artifact Review and Badging
  - https://www.acm.org/publications/policies/artifact-review-and-badging-current
- FAIR Guiding Principles
  - https://doi.org/10.1038/sdata.2016.18
- NeurIPS Paper Checklist
  - https://nips.cc/public/guides/PaperChecklist
- REFORMS
  - https://reforms.cs.princeton.edu/

## Default Placement In Workflow

- repo-wide review で research / experiment scope を含む場合に使います。
- `Research-Driven Change` で methodology、benchmark protocol、artifact policy、reporting policy を大きく変える場合に使います。
- `project-review` と `comprehensive-review` の research-specific extension として扱います。

## Boundary

- 単一 run の批判的レビューは `agents/skills/critical-review.md` を使います。
- user-facing report 単体のレビューは `agents/skills/report-review.md` を使います。
- 研究全体の反復設計は `agents/skills/research-workflow.md` を使います。

## Implementation Surface

- `.github/skills/25-research-perspective-review/`
