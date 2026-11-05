# Oneiros

在一个封闭的人工生命沙箱里，设计者只给出身体、世界、学习规则的形式和选择方式，然后观察智能与类人情绪能否从这些条件中涌现。

可观测不是目的，智能与情绪的涌现才是。

## 文档

| 文档 | 作用 |
|---|---|
| [docs/PRINCIPLES.md](docs/PRINCIPLES.md) | 核心问题、唯一硬规则、假设 H0–H6、判据、方法规则 |
| [docs/SANDBOX.md](docs/SANDBOX.md) | 沙箱规格：ECS、世界、身体、大脑、观测、干预、里程碑 |
| [docs/ARCHITECTURES.md](docs/ARCHITECTURES.md) | 六个候选大脑 B0–B5，记忆与睡眠机制 |
| [docs/EMOTIONS.md](docs/EMOTIONS.md) | 只用于分析的情绪与行为解释库，代码不得引用 |

## 技术选型

JAX + 纯函数 pytree ECS（SANDBOX.md §2–3）：状态是 pytree，快照即保存、分叉即复制、回放即重跑；随机数按 (world_key, tick, system) 派生，观测器拿不到 key。

旧草稿里的 "JAX + Esper" 已作废：Esper 是可变的 Python ECS，与确定性 pytree 快照不兼容。

## 开发

~~~sh
uv sync                             # 若 ~/.cache/uv 不可写：UV_CACHE_DIR=$PWD/.uv-cache uv sync
uv run pytest
uv run life-sandbox --ticks 200
~~~

## 状态

M0 引擎核心进行中。当前已落地：确定性 key 派生与状态哈希、ECS 组件与槽位、事件缓冲、调度器骨架、世界初始化、配置系统。
