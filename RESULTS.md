# Oneiros 结果记录

每条结论一行。状态取 {确认, 探索, 作废}；run_id 指向 runs 表（SANDBOX.md §7.2）；
prereg_id 指向 prereg/ 下的预注册文档。探索性结果不能作为确认引用；作废的结果保留并标注作废，不删除（SANDBOX.md §10）。

| 日期 | 结论 | 状态 | run_id | prereg_id | 备注 |
|---|---|---|---|---|---|
| 2026-11-06 | M1 探索：宽松搜索区（代谢 1/20、灌木 2–4×、温差 ≤0.5）可自持但不属于设计值世界 | 探索 | round2feas / round2（远端 runs/） | — | results/001-m1-exploratory.md |
| 2026-11-06 | M1 归因修正后：设计值世界死因 99.4–99.95% 为营养（动物物质缺失占优），热/冷 0、咬 0.5%、毒≈0 | 探索 | diagnostics2（远端 runs/） | — | results/002-m1-attribution.md |
| 2026-11-06 | M1 可行性：关闭重播种后设计值世界灭绝（fixed/control 种群均值≈0.5、探针≈1.7）；净繁殖率<1，营养死因≈100%，出生失败非主因；创始动物 logit−1 仅小幅提高出生 | 探索 | diagnostics3/4/5/6（远端 runs/） | — | results/003-m1-viability.md |
| 2026-11-06 | M1 决定性可行性：库存感知探针+关闭救援，设计代谢 0.002 灭绝；0.5× 仍灭绝；0.25× 32k tick 内 2/4 世界灭绝、2 个低种群振荡；harvest/cost 1.03–1.09、排泄占吸收 30–31%、动物营养主限制；fixed 臂在 0.25× 仍灭绝 | 探索 | diagnostics7/8/9/10（远端 runs/） | — | results/004-m1-decisive.md |
| 2026-11-06 | M1 探针 v2 矩阵：frozenstart 世界 6 组均 R0≈1.0（种群 26–101），target 世界 6 组全部灭绝（R0all 0.06–0.56，远低于模型预测 4–6）→ 按复核判读进入“对账”，暂不改规格常量；N 支付已生效，B 支付对照可用 | 探索 | matrix2（远端 runs/） | — | results/005-m1-probe-v2-matrix.md |
| 2026-11-06 | M1 静止食物对账：钉库存 + 匹配食性时父代 R0=5（模型预测 4–6），正常再生单丛 R0=2、摄入 0.006<需求 0.007 → 引擎预算对账通过，matrix2 差距归因于觅食效率；下一步扫描 base_metabolism | 探索 | stationary（远端 runs/） | — | results/006-m1-stationary-reconciliation.md |
| 2026-11-06 | M1 R0 修正：按（世界,uid）分组、分母含创始×世界；target 修正 R0 0.02–0.24；32 个体静止对账创始平均 R0=2.16（12/32 零胎）、no_reserve 卡住营养全为 animal；降级探针校准为饥饿阈值 h=0.2（摄入 56.7%）；修正后 base 扫描重跑中 | 探索 | stationary32 / degraded / basescan2（远端 runs/） | — | results/007-m1-r0-fix.md |
| 2026-11-06 | M1 探针索引修复：comp(种类,营养) 索引错误改正，选种迁 analysis/probe.py + 7 单测；修复后 target 仍全灭绝（繁殖出生 R0 0.20–0.58，摄入 0.0086–0.0092），frozenstart 种群 40–126（繁殖出生 R0 0.93–0.97）；降级 h=0.2 摄入为 v2 的 58.6%；base 扫描 16 世界重跑中 | 探索 | matrix3 / degraded2 / basescan3（远端 runs/） | — | results/008-m1-probe-fix-matrix.md |
| 2026-11-07 | M1 probe v3：覆盖率选种（不用 c_tot）、stock-min 两档、logit −1 默认、R0 出生世代截断、摄入按成熟度拆分；f16 矩阵繁殖出生 R0 0.42–0.83（base 0.6 最好，存活 0–8/16），降级臂 0 胎；f4 稀疏验证运行中 | 探索 | matrix4 / matrix4_f4（远端 runs/） | — | results/009-m1-probe-v3-matrix.md |
