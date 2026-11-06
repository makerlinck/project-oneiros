# Oneiros 结果记录

每条结论一行。状态取 {确认, 探索, 作废}；run_id 指向 runs 表（SANDBOX.md §7.2）；
prereg_id 指向 prereg/ 下的预注册文档。探索性结果不能作为确认引用；作废的结果保留并标注作废，不删除（SANDBOX.md §10）。

| 日期 | 结论 | 状态 | run_id | prereg_id | 备注 |
|---|---|---|---|---|---|
| 2026-11-06 | M1 探索：宽松搜索区（代谢 1/20、灌木 2–4×、温差 ≤0.5）可自持但不属于设计值世界 | 探索 | round2feas / round2（远端 runs/） | — | results/001-m1-exploratory.md |
| 2026-11-06 | M1 归因修正后：设计值世界死因 99.4–99.95% 为营养（动物物质缺失占优），热/冷 0、咬 0.5%、毒≈0 | 探索 | diagnostics2（远端 runs/） | — | results/002-m1-attribution.md |
| 2026-11-06 | M1 可行性：关闭重播种后设计值世界灭绝（fixed/control 种群均值≈0.5、探针≈1.7）；净繁殖率<1，营养死因≈100%，出生失败非主因；创始动物 logit−1 仅小幅提高出生 | 探索 | diagnostics3/4/5/6（远端 runs/） | — | results/003-m1-viability.md |
| 2026-11-06 | M1 决定性可行性：库存感知探针+关闭救援，设计代谢 0.002 灭绝；0.5× 仍灭绝；0.25× 32k tick 内 2/4 世界灭绝、2 个低种群振荡；harvest/cost 1.03–1.09、排泄占吸收 30–31%、动物营养主限制；fixed 臂在 0.25× 仍灭绝 | 探索 | diagnostics7/8/9/10（远端 runs/） | — | results/004-m1-decisive.md |
