# 单卡 LIBERO FPO 复现记录

## 版本与对照

基线为 FPO `86f2c9cb257a65186d425cddd88b635dc6e9a4a8`，OpenPI 子模块为
`a6d2400d2534ce32e7bdf8747709b97aaef8ec04`，LIBERO 子模块为
`87edbd1b823e464bc66557d213cc45e9f88aa543`。AutoDL 上对已有 LIBERO 素材逐文件核验
Git blob SHA，复用 1,025 个一致文件，从固定提交补入 1 个不同文件。复核了全部 1,026 个
LIBERO 运行文件。OpenPI 使用固定提交的归档。原论文入口保留。

任务为 LIBERO-90 task 56。训练使用初态 ID 0–5，评测使用 10–13。所有评测臂使用同一
BDDL、控制频率、停止谓词、400 控制步上限及配对种子。每回合先恢复固定状态并执行
10 步开爪等待，经初态及初始观测 SHA 校验后比较。原训练入口使用环境随机 reset。
等待步数单独规定，表中控制步统计从策略接管开始。

| 配置 | π₀ 噪声输入 | Latent Actor |
| --- | --- | --- |
| B0 | 每次采样独立的 50×32 标准高斯 | 无 |
| B1 | 一个 1×32 latent 重复到 50×32 | 与 B2 同初始化，零更新 |
| B2 | 与 B1 相同 | 完成规定的在线更新 |

B1/B2 同条件、同决策序号复用 Actor 随机种子。B0 使用相同种子方案的全形状采样。
评测同时记录 B2 对 B0、B1 的配对获失。相同控制步只标记时间基准，回合的物理过程会
随策略分叉。结果只能解释这组初态和训练预算。

## 论文与源码审计

- 流模型的精确似然比需要额外求解概率流与散度。FPO 用旧损失减当前损失的
  CFM 差值做代理，标准化后映射为 `exp(βz)`，再代入 clipped surrogate。该映射依赖
  CFM 局部排序与密度变化的相关性。同一样本会随所在 minibatch 的均值和标准差变动。
  [论文 §III](https://arxiv.org/html/2510.09976v2#S3)
- 采集时在 `sync_old_actor` 后，对实际供 π₀ 解码的已截断 latent 缓存同样的
  `eps`、`t` 和旧 CFM 损失。更新时用同一未增广视觉输入及随机采样重新计算 CFM。
  代码切断批统计量、旧损失、advantage 的梯度，当前 CFM 仍有梯度。
  论文中关于切断 ratio 梯度的文字按上游实现说明理解，完全切断 ratio 会令该 Actor
  目标失去梯度。 [上游代码对照](https://github.com/tuotuoLyu/FPO/blob/86f2c9cb257a65186d425cddd88b635dc6e9a4a8/docs/PAPER_ALIGNMENT.md)
- 10 个 Q head 使用独立于 Actor 的视觉编码器。TD target 使用当前 Actor 的新 latent、
  target Q 的最小值和终止 mask，Polyak 系数 0.005。GAE 使用旧数据的实际 latent，
  超时时在最后观测上采样新 latent 来 bootstrap。同一折扣 0.99 应用在每次模型决策。
  上游 Replay 的 `discount` 列写入了 `γ^query_freq`，当前 FPO 目标函数不读取该列。
- **已确认的计数问题。** 上游用 Replay 条数控制首轮更新，采集函数却用累积梯度
  步判断是否仍处于暖启动。达到 Replay 门槛后，训练已经更新但采集仍可能继续走
  Gaussian。本分支独立入口用同一个 Replay 门槛决定两处行为，并测试边界。
  Gaussian 暖启动样本仍会按原算法缓存 rollout Actor 的 CFM，使用这些样本的比率
  代理不能视为严格的 importance ratio。
- **工程差异。** 上游采样窗口只限制可抽样范围，数组继续增长。原 checkpoint
  含 learner/optimizer，却缺少 Replay、随机流和循环游标。独立入口保存上述状态，
  完成评测回合作为不可覆盖的记录。回合中断重做当前回合，更新中断从最近提交状态
  续跑。原版例程是否能安全承载 500,000 次更新与长 Replay，留待实际压力测试。
- **非有限值检查。** 上游有 Actor 输入跳步和梯度清理。独立入口检查非有限指标、
  跳步标志及更新前后参数差，异常会停止。已准备 GPU 探针测量真实参数更新，
  联合驻留 π₀ 的峰值仍由完整流程测量。

## 第一轮预算

保留图像分辨率 128、ResNet-34、10 Q、4 次 CFM 采样、6 步 Euler、32 步执行窗口、
GAE 与损失定义。缩放量集中在
[`configs/libero_4090_p1.json`](../configs/libero_4090_p1.json)。

| 变量 | 论文脚本 | 单卡 P1 |
| --- | ---: | ---: |
| Batch | 210 | 8 |
| 预热 Replay 条数 | 1,000 | 16 |
| 每新 Replay 条数的更新预算 | 40 | 2 |
| 梯度更新上限 | 500,000 | 48 |
| GAE 计算分块 | 256 | 8 |
| 评测初态数／臂 | 10 个回合 | 4 个独立固定初态 |

P1 至少采集一个已更新 Actor 的训练回合，并至少用它参与后续更新。训练语义保留
上游每回合重新计算 GAE、episode-wise old Actor 的组织。P1 用来确认在线学习链路和
测量吞吐。四个配对初态只适合生成下一轮假设。后续扩量须依据实测吞吐与配对结果，
首轮不自动发起 P2。

## 执行和恢复

使用专用运行目录。运行器要求同一张 GPU 没有其他计算进程、固定源码校验标识、
本地 tokenizer 及 NVIDIA EGL 配置。进入 P1 还要求官方 GCS checkpoint 的完整校验回执。
采用 pinned OpenPI 的本地权重加载接口。下载端核对官方对象版本、CRC32C、MD5
（官方提供时）及 SHA256，运行端复核所有权重和归一化文件的 SHA256。

```bash
cd "$FPO_RUNTIME/source"
scripts/run_libero_4090.sh probe "$FPO_RUNTIME" "$FPO_RUNTIME/runs/probe"
scripts/run_libero_4090.sh run "$FPO_RUNTIME" "$FPO_RUNTIME/runs/p1" --max-seconds 9000
cat "$FPO_RUNTIME/runs/p1/status.json"
touch "$FPO_RUNTIME/runs/p1/STOP"
# 确认状态为 paused、旧进程退出后，移走本次任务的暂停标识再恢复。
mv "$FPO_RUNTIME/runs/p1/STOP" "$FPO_RUNTIME/runs/p1/STOP.previous"
scripts/run_libero_4090.sh run "$FPO_RUNTIME" "$FPO_RUNTIME/runs/p1" --resume --max-seconds 9000
```

`--max-seconds` 是单次调用预算，在回合与更新边界检查。恢复前应从首轮三小时总预算扣除
已用时长。记录的控制步数、π₀ 决策次数、Replay 条数、梯度步数分别计数。首调用和首更新
包含 JAX 编译，逐回合和逐更新时间单独保留。资源 CSV 每 2 秒采样，峰值是采样下界。
结果目录包含各回合视频、对应动作与 latent、配置、初态和 Checkpoint 校验信息。
该目录位于实验运行区，仓库仅提交程序、配置、测试和汇总证据。

## 阶段证据

- 上游 CPU 玩具脚本结束于 `SMOKE OK`。上游测试替身的循环为 11 回合、40 Replay
  条、60 次更新，结束于 `LOOP CHECK OK`。
- AutoDL 隔离环境已用 JAX CPU 执行全部 21 项上游测试。包含真实 JAX Actor/Critic
  测试、CFM、Euler 和 Replay/GAE。环境为 Python 3.11、JAX 0.5.3、Flax 0.10.2、
  Orbax 0.11.13、MuJoCo 3.3.1、robosuite 1.4.1。JAX 与 Orbax 版本和论文脚本
  依赖文件有差异。
- 独立入口增加初态拆分、动作块窗口、checkpoint 完整性及暖启动边界测试。
- π₀ 官方模型总计 19 个对象、约 11.19 GiB。上游入口中的 S3 地址返回 404，
  当前经官方公开 GCS 分段下载，π₀ 权重加载、B0/B1/B2 和真实在线 RL 成绩需等验收。
  [OpenPI 官方权重说明](https://github.com/Physical-Intelligence/openpi#model-checkpoints)
