# Workflow References

この文書は、workflow、review、agent system、report policy を設計するときに参照した外部資料の索引です。
参考文献そのものは `references/` に置き、この文書は「何を根拠にどの方針を作ったか」を辿るための正本にします。

workflow や review 観点を外部根拠で更新した場合は、この文書に出典を追記しなければなりません。

## Agent Runtime And Customization

- [Custom instructions with AGENTS.md - Codex | OpenAI Developers](https://developers.openai.com/codex/guides/agents-md)
  - root `AGENTS.md` を入口にする運用の根拠です。
- [Subagents - Codex | OpenAI Developers](https://developers.openai.com/codex/subagents)
  - Codex subagent の置き方と使い分けの根拠です。
- [How Claude remembers your project - Claude Code Docs](https://code.claude.com/docs/en/memory)
  - `CLAUDE.md` を薄い adapter にする判断の参考です。
- [Copilot customization cheat sheet - GitHub Docs](https://docs.github.com/en/copilot/reference/customization-cheat-sheet)
  - GitHub Copilot 用 adapter と custom instructions の整理に使った資料です。
- [Your first custom instructions - GitHub Docs](https://docs.github.com/en/copilot/tutorials/customization-library/custom-instructions/your-first-custom-instructions)
  - Copilot 側の最小入口設計の参考です。

## System Development, Security, Release, And Operations

- [NIST SP 800-218, Secure Software Development Framework (SSDF)](https://csrc.nist.gov/pubs/sp/800/218/final)
  - secure development workflow、review gate、supply-chain 観点の根拠です。
- [Microsoft Security Development Lifecycle](https://www.microsoft.com/en-us/securityengineering/sdl)
  - security review と設計段階の gate を考えるときの基礎資料です。
- [OWASP Threat Modeling Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Threat_Modeling_Cheat_Sheet.html)
  - threat-model workflow を追加するときの根拠です。
- [Google SRE: Release Engineering](https://sre.google/sre-book/release-engineering/)
  - release-readiness と deploy 前後の運用手順の根拠です。
- [Google SRE: Emergency Response](https://sre.google/sre-book/emergency-response/)
  - incident-response や hotfix workflow の根拠です。
- [Google SRE Workbook: Postmortem Culture](https://sre.google/workbook/postmortem-culture/)
  - postmortem と failure follow-up workflow の根拠です。
- [Microsoft Azure Well-Architected: Safe deployment practices](https://learn.microsoft.com/en-us/azure/well-architected/operational-excellence/safe-deployments)
  - safe-deployment と rollback readiness の観点を入れる根拠です。

## Research, Experiment, And Reporting

- [Experiment Workflow References](/workspace/references/experiment_workflow/README.md)
  - 実験 workflow、再現性、benchmark、figure、批判的レビューの local index です。
- [Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research](/workspace/references/experiment_workflow/Sandve_2013_Ten_Simple_Rules_for_Reproducible_Computational_Research.pdf)
  - reproducibility review の基礎です。
- [Wilson et al. (2014), Best Practices for Scientific Computing](/workspace/references/experiment_workflow/Wilson_2014_Best_Practices_for_Scientific_Computing.pdf)
  - scientific-computing review の基礎です。
- [Wilson et al. (2017), Good Enough Practices in Scientific Computing](/workspace/references/experiment_workflow/Wilson_2017_Good_Enough_Practices_in_Scientific_Computing.pdf)
  - 軽量な研究運用と実務レベルの checklist の根拠です。
- [Nature, Guidance on Reproducibility for Papers Using Computational Tools](/workspace/references/experiment_workflow/Nature_Guidance_on_Reproducibility_for_Papers_Using_Computational_Tools.pdf)
  - computational paper の再現性要件を整理するときに使った資料です。
- [Minocher et al. (2023), Implementing Code Review in the Scientific Workflow](/workspace/references/experiment_workflow/Minocher_2023_Implementing_Code_Review_in_the_Scientific_Workflow.html)
  - scientific workflow に code review を組み込む根拠です。
- [Rougier et al. (2014), Ten Simple Rules for Better Figures](/workspace/references/experiment_workflow/Rougier_2014_Ten_Simple_Rules_for_Better_Figures.pdf)
  - report-review の figure / table 観点の基礎です。
- [Bartz-Beielstein et al. (2020), Benchmarking in Optimization: Best Practice and Open Issues](/workspace/references/experiment_workflow/Bartz-Beielstein_2020_Benchmarking_in_Optimization_Best_Practice_and_Open_Issues.pdf)
  - benchmark fairness と比較条件の根拠です。
- [Benchmarking Crimes: An Emerging Threat in Systems Security](https://arxiv.org/abs/1801.02381)
  - benchmarking anti-pattern review の根拠です。
- [Artifact Review and Badging - Current | ACM](https://www.acm.org/publications/policies/artifact-review-and-badging-current)
  - artifact readiness と公開物 completeness の根拠です。
- [The FAIR Guiding Principles for scientific data management and stewardship](https://www.nature.com/articles/sdata201618)
  - FAIR-data review の根拠です。
- [NeurIPS Paper Checklist](https://nips.cc/public/guides/PaperChecklist)
  - ML 系 report と experiment checklist の根拠です。
- [REFORMS: Consensus-based Recommendations for Machine-learning-based Science](https://reforms.cs.princeton.edu/)
  - ML-science reporting review の根拠です。

## Related Reference Collections

- [references/README.md](/workspace/references/README.md)
  - reference 全体の入口です。
- [Generative AI References](/workspace/references/generative_ai/README.md)
  - LLM を使った review、literature research、scientific workflow 支援の資料です。
- [Sparse Grid References](/workspace/references/sparse_grid/README.md)
  - Smolyak、sparse-grid、combination technique の研究資料です。
