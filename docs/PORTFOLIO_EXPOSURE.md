# 多策略持仓重叠与 ETF 穿透证据

`portfolio-overlap`读取调用方提供的策略 NAV 权重和 QDK 披露持仓，生成可供 Studio 或
ReportHub 只读消费的 JSON 报告。它不访问网络、不下载成分，也不修改风险限制、账户、持仓
或前向预测。

```sh
quant-risk portfolio-overlap \
  --input examples/portfolio_overlap_study.json \
  --out runs/portfolio-overlap.json
```

输入与证据有效且报告成功写出时退出 0，即使某一 pair 因覆盖不足而标记`unavailable`；输入、
守恒、schema 或风险证据核验失败时退出 2 且不生成输出。输出路径必须不存在，命令不会覆盖
旧报告。

可选的`--risk-run runs/risk-example`会先调用现有`verify_risk_validation`，校验固定文件集合、
哈希和原输入完整重算，再把已有组合及主动风险评分整理为可读摘要。摘要不重新评分、不调参，
`complete`也不表示校准通过。风险证据被篡改时不会生成组合报告。

输入和输出分别遵循：

- [`portfolio-exposure-study-v1.schema.json`](../schemas/portfolio-exposure-study-v1.schema.json)
- [`portfolio-exposure-report-v1.schema.json`](../schemas/portfolio-exposure-report-v1.schema.json)

## 时间和估值契约

`business_at`是报告业务时点，`knowledge_at`是资料获知截止时点，且不得晚于业务时点。
策略快照的`effective_at`不得晚于业务时点，`available_at`不得晚于获知时点，也不得早于
自身有效时点。基金披露先按`available_at <= knowledge_at`过滤，再由 QDK 在业务时点选择
最新披露并按`holding_date`计算陈旧天数。因此获知截止后的成分不会进入暴露；若它是唯一
可见候选，相关 gross 权重进入`future`覆盖桶。

每个 position 的`weight`必须已经由调用方按统一`base_currency`估值，并除以该策略 NAV。
本软件不接收 shares、amount 或 market value，不获取或认证价格、FX、NAV 分母和真实持仓。
`base_currency`只是调用方声明的估值口径，不是已认证换汇结论。缺少 currency、权重、来源或
时间会使输入不可用；不会补零。权重不净和为 1 时不会推断残余现金。

权重允许为负且不限制 gross：负证券或基金表示空头，负现金表示借款。基金披露内部继续沿用
QDK 的 long-only、单快照权重不超过 100% 契约；空 ETF 通过单位穿透后整体乘负号。报告分别
展示 signed net、absolute gross、穿透路径 gross、底层证券聚合后的 gross 和内部净额抵消，
不会用净敞口冒充总敞口。

## 穿透和覆盖

同一策略的重复 position 先按 instrument、asset type 和 currency 聚合。同一底层证券由直接
持仓和任意 ETF 路径贡献时，再按稳定 instrument identity 与 currency 聚合一次。QDK 负责
多层递归、完整披露快照选择、未披露残量、陈旧资料、循环、深度限制和路径权重守恒。

coverage 分母始终是输入路径的 absolute gross，分桶互斥且相加等于分母：

- `covered`：直接持仓或可用披露的底层证券/现金；
- `unknown`：没有披露或披露权重未满 100% 的残量；
- `stale`：最新可用披露超过`max_age_days`；
- `future`：只有在获知截止后或业务日期后的披露；
- `cycle`、`depth_limit`：被显式终止的递归路径。

每个策略快照、披露集合、实际选中披露和底层来源都有 SHA-256。哈希绑定所给内容，但不能
认证外部提供者身份，也不能证明合成数据是真实市场资料。

## 重叠定义和手算示例

pairwise 只比较双方已聚合的非现金 security 暴露。对于共同证券`i`：

`shared_i = min(abs(w_left,i), abs(w_right,i))`

同号计入`aligned`，异号计入`opposing`。完整 overlap ratio 为共同 gross 之和除以双方较小的
已知 security gross。它是持仓重叠比例，不是收益相关性、风险预测或分散化结论。

合成示例中 alpha 直接持有 A 20%，并持有 ETF 60%。ETF 披露 A/B/现金为 50%/40%/10%，
所以 alpha 底层为 A 50%、B 24%、现金 26%。beta 为 A 25%、B 25%、现金 50%。证券共同
gross 为`min(50%,25%)+min(24%,25%)=49%`，较小证券 gross 为 50%，重叠比例为 98%。
该例只验证算术和契约，不是市场认证。

只要任一策略存在 unknown、stale、future、cycle 或 depth_limit，pair `status`就是
`unavailable`，完整`overlap_ratio_of_smaller_gross`为 null。程序仍保留明确命名的
`diagnostic_known_only_ratio`和已知共同证券，供定位资料缺口；该诊断不能显示为完整通过或
用于宣称两个策略可比。现金-only 或没有非现金证券分母也不可比较。

## 风险摘要边界

`validate-forecasts`和`verify-forecasts`现在直接打印`quant-risk.forecast-readable-summary/v1`：
组合和主动序列的预测/实际年化波动、bias statistic、QLIKE、零方差状态，以及输入、预测和
成熟结果哈希。原生预测生成、因果时点过滤、成熟收益匹配和评分公式完全保留。预测期初资料
与随后成熟收益仍由现有风险验证器隔离；组合重叠报告不会把风险评分写回风险限制或账户。

