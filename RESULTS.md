# Oneiros 结果记录

每条结论一行。状态取 {确认, 探索, 作废}；run_id 指向 runs 表（SANDBOX.md §7.2）；
prereg_id 指向 prereg/ 下的预注册文档。探索性结果不能作为确认引用；作废的结果保留并标注作废，不删除（SANDBOX.md §10）。

| 日期 | 结论 | 状态 | run_id | prereg_id | 备注 |
|---|---|---|---|---|---|
| 2026-11-06 | M1 探索：宽松搜索区（代谢 1/20、灌木 2–4×、温差 ≤0.5）可自持但不属于设计值世界；设计值世界下 fixed 与随机基因组对照无差别（均靠重播种），温度与毒素是主要死因；脚本觅食者在设计值世界也无法自持 | 探索 | round2feas / round2 / diagnostics（远端 runs/） | — | results/001-m1-exploratory.md |
