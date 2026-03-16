# Smolyak Integrator Theme

## Scope

このページは、Smolyak 積分器について複数 branch と実験から得た知見を、再利用しやすい形でまとめた theme note である。詳細な経緯や日付順の判断は report や worktree note に譲り、ここでは「いま何が言えるか」を topic-first で整理する。

## Known

現在の知見として最も確かなのは、旧版の explicit-grid 型実装では host 側の grid 構築と重複統合が主要コストになっていたということである。point 配列、weight 配列、term ごとの一時 tensor、`np.unique(axis=0)` を含む大域集約が CPU RSS と pinned host memory を強く圧迫し、GPU 利用率が低く見える原因にもなっていた。

tuned 版では explicit grid 依存を減らし、`initialize_smolyak_integrator(...)` と `integrate(f, integrator)` を中心に据える設計へ寄せたが、それで問題が完全に消えたわけではない。少なくとも `level=1` の partial では、`num_points=1` の時点ですでに `integrator_init_seconds` と `process_rss_mb` が次元に対して急増している。したがって現在の主要制約は、積分点数よりも初期化、lowering、compile、あるいはそれに付随する metadata 構築にあると考えてよい。

HLO 解析でも同じ傾向が見えている。小さい case でも `stablehlo.while`, `func.call`, `stablehlo.gather` が目立ち、算術 kernel が主役になっている形ではない。これは tuned 実装が「explicit grid を減らした」ことと、「GPU 算術が支配的になった」ことが同義ではないと示している。

## Worked

公開 API を `integrate(f, integrator)` に寄せ、explicit grid を返す経路を外したことは有効だった。これにより、積分器の責務が「点集合を外へ見せる」ことから「積分を実行する」ことへ整理され、内部表現を rule table と term plan に寄せやすくなった。また、experiment runner を `jax_util.experiment_runner` に切り出したことで、JSONL 逐次保存、child completion の明示通知、multi-GPU の slot 管理が実験ごとに再利用できるようになった。

長時間実験の運用では、JSONL 逐次保存と final JSON の `main` への持ち帰りも有効だった。これにより、途中停止しても partial を集計し直せるし、後から別の図を再生成することもできる。

## Coding Pattern

この topic で効いた coding pattern は、公開 API と内部表現を分けること、runner と experiment 固有ロジックを分けること、そして note を self-contained に保つことである。特に experiment runner では、host が free slot を管理し、child が JSONL 追記と completion record を担当する形が扱いやすかった。失敗分類も、`oom`, `worker_terminated`, `error` を分けて記録しておくと、partial の段階でも結果が読みやすくなる。

また、長時間実験の case ordering は coding detail に見えて、結果解釈に直結する。`dtype -> level -> dimension` では途中停止時に比較が崩れやすく、`dimension -> level -> dtype` のほうが frontier を読みやすかった。実験コードは「何を測るか」だけでなく、「どの順に落ちても何が読めるか」まで含めて設計すべきである。

## Did Not Work

explicit grid を前提にした大域統合は、少なくともこの project の規模ではうまくいかなかった。`np.unique(axis=0)` を含む host 側集約は、数値計算そのものより前にメモリと初期化時間の問題を引き起こした。また、tuned 版でも「explicit grid を減らしたから GPU が主役になる」とはならず、代わりに initializaion-bound な挙動が前面に出た。

実験運用では、`dtype -> level -> dimension` 順の case 配列もよくなかった。途中停止すると先頭 dtype だけが大量に残り、比較用データとして偏ってしまう。さらに、`main` に note だけ残して raw JSON 相当を持ち帰らない運用も不十分だった。後から別の可視化や集計をやり直すためには、少なくとも `cases` 配列を含む final JSON を `main` に持ち帰る必要がある。

## Pitfall

「GPU が使われていないように見える」ことと「GPU 分離が壊れている」ことは同じではない。今回のケースでは、GPU 1,2 が idle に見える場面があっても、child ごとの GPU 可視性分離自体は成立していた。問題は runner の GPU 割当ではなく、CPU 側初期化が長く、GPU 実行が短く観測しにくいことだった。このように、観測上の症状と原因を直接結び付けないことが重要である。

## Likely

実験 runner 側の問題は、以前よりかなり整理されたとみてよい。child ごとの GPU 可視性は切り分け済みで、各 child が `CUDA_VISIBLE_DEVICES` に応じて 1 GPU だけを見ていることは確認されている。したがって、GPU 1,2 が遊んで見える現象の主因は GPU 分離バグではなく、CPU 側初期化が長いこと、そして GPU 実行が短く観測しにくいことだとみるのが自然である。

case ordering は結果の読みやすさに大きく効く。`dtype -> level -> dimension` は途中停止時に比較が崩れやすく、`dimension -> level -> dtype` のほうが frontier を早く読みやすい。これは Smolyak 固有の数学ではなく、長時間実験を設計するうえでの実務的知見である。

## Open

まだ未解決なのは、tuned 版の初期化コストの正体をどこまで分離できるかである。現在は `integrator_init_seconds` という大きな塊として観測しているが、その中に

- rule 準備
- tracing
- lowering
- compilation
- device まわりの初期化

のどれがどの程度入っているかは、さらに切り分けが必要である。また、高 level における dtype ごとの数値精度比較は、現時点では initialization-bound な挙動に隠れて十分に観測できていない。

## Practical Guidance

Smolyak 積分器を今後改善するときは、数値精度の議論だけでなく、保持戦略と初期化経路の議論を常に分けるべきである。旧版の主問題は explicit grid と host memory であり、tuned 版の主問題は初期化・lowering・compile 側に移っている。したがって、次の改善は「旧版より tuned 版が優れているか」ではなく、「tuned 版の initialization-bound をどこまで崩せるか」を基準に評価するのがよい。

実験運用としては、JSONL 逐次保存、child completion の明示通知、case ordering の明確化、そして `main` 側への final JSON 持ち帰りが重要である。これにより、途中停止しても知見が失われず、あとから別の図や集計を再生成できる。

## References

- [Smolyak Integrator Tuning Report](/workspace/notes/experiments/smolyak_integrator_report.md)
- [Legacy Smolyak Results](/workspace/notes/experiments/legacy_smolyak_results_20260316.md)
- [Tuned Smolyak Partial Results](/workspace/notes/experiments/tuned_smolyak_partial_results_20260316.md)
- [Smolyak Tuning Worktree Extraction](/workspace/notes/worktrees/worktree_smolyak_tuning_2026-03-16.md)
- [Experiment Runner Modularization Worktree Extraction](/workspace/notes/worktrees/worktree_experiment_runner_module_2026-03-16.md)
