# Barra风格统计风险模型

`quant_risk_monitor.factor_model`实现一个可审查的线性因子风险模型最小闭环。它借用
`Σ=XFXᵀ+D`这一类Barra风格结构，但不是MSCI Barra模型，不包含MSCI的专有数据、
描述子、因子定义、估计方法或校准结果。

调用方必须明确选择`model_kind`：

- `fundamental_style`表示调用方提供了有来源证据的基本面、行业或风格暴露。本库只验证
  输入和估计过程，不认证暴露的经济含义。
- `statistical_proxy`表示暴露全部或部分来自收益、价格或其他统计代理。模型、诊断和
  组合风险输出始终保留`is_proxy=true`，不得将其展示为完整基本面Barra模型。

## PIT输入契约

公开构造器和估计入口为：

```python
ExposureSnapshot(*, effective_at, available_at, values, source="")
AssetReturnObservation(*, period_start, period_end, available_at, values, source="")
FactorModelConfig(
    *,
    model_kind,
    min_periods=20,
    min_assets_per_period=6,
    min_asset_observations=20,
    covariance_shrinkage=0.2,
    specific_variance_shrinkage=0.2,
    annualization=252,
    max_condition_number=1e8,
)
fit_barra_style_risk_model(
    *, exposure_snapshots, return_observations, as_of, config
)
```

所有时间必须带时区，内部统一为UTC。对结束于`t`的收益截面，估计器只选择在
`period_start`之前已经生效且已经可得的最新暴露，即同时满足
`effective_at<=period_start`和`available_at<=period_start`。收益还必须满足
`period_end<=as_of`和`available_at<=as_of`，因此未成熟或当时不可得的收益不能进入
估计窗口。

每个暴露快照必须是完整的`asset×factor`有限数值矩阵，每个收益观察必须是完整的
资产截面。资产集合改变、因子集合改变、重复时段、重叠收益时段、未来输入、缺失值和
非有限值都会被拒绝。库不会用0填补缺失暴露，也不会自动增加截距、market或行业因子。
真实数值0是合法暴露。若需要共同市场因子，调用方应显式提供常数`market`列并保留其
来源和代理属性。

## 估计和验证

对每个已成熟收益期，模型计算普通最小二乘横截面回归：

`r_t = X_t f_t + ε_t`

输入资产数必须达到`min_assets_per_period`且严格多于因子数。`X_t`必须满列秩，条件数
不得超过`max_condition_number`。估计器不会删除共线因子或伪造替代因子；这些问题会
阻断模型构建并返回带时段的错误。

因子收益样本协方差向其对角矩阵收缩，再对纯数值误差产生的负特征值裁剪至0，得到
年化PSD矩阵`F`。每个当前资产必须至少有`min_asset_observations`个特异收益；资产特异
方差向当前资产的平均特异方差收缩，得到对角`D`。最终矩阵严格按
`Σ=XFXᵀ+D`重建。收缩系数由配置显式给出，本实现不声称它们是供应商校准值。

模型在内部保存矩阵的独立副本，`exposures`、`factor_returns`、`factor_covariance`、
`specific_variances`和`asset_covariance`属性返回防御性副本。每次分析或序列化前都会
重新验证标签、有限值、PSD性质以及`Σ=XFXᵀ+D`恒等式；若内部状态被非正常方式修改，
模型会显式拒绝，而不会临时裁剪或重建矩阵来掩盖不一致。

`fit_barra_style_risk_model`仍是等权普通最小二乘的最小闭环。市值加权、行业约束、
指数加权协方差、序列相关修正、波动率机制缩放和特异风险的结构化混合在
`fit_equity_style_risk_model`里，参数由`EquityRiskConfig`显式给出。

## 股票风格估计

`quant_risk_monitor.equity_style.fit_equity_style_risk_model`在同一套
`Σ=XFXᵀ+D`和时点规则上补上股票风险模型常用的估计步骤。它仍然不是MSCI
Barra：半衰期、滞后阶数、收缩强度、极值倍数和描述子混合权重都是调用方配置，
诊断里的`parameters_are_vendor_calibration`恒为false。

输入分成三组快照，外加已经实现的收益：

- `StyleExposureSnapshot`是风格描述子。单元格可以缺失，但不能是无穷大。
- `IndustrySnapshot`是调用方自己的行业标签。模型据此生成国家因子和行业哑变量。
- `MarketCapSnapshot`是正的市值，用于回归权重和行业约束。

对每一期收益，估计器只使用`period_start`之前已经生效且已经可得的快照。覆盖股票池是当期风格、行业和正市值的交集。缺失风格按`imputation`处理：`reject`直接失败，`industry_median`用行业中位数，仍缺再退到全市场中位数，不用0填充。`standardize_styles=True`时，每个风格先按等权均值和标准差去极值，再按市值加权标准化到均值0、标准差1，然后按配置把目标列对控制列做市值加权正交。默认正交只在列名已经存在时生效，例如`nonlinear_size`对`size_log_mcap`，`residual_vol_252d`对贝塔和规模，`long_term_reversal_504_252`对动量。

因子收益用加权最小二乘。`weight_kind="sqrt_market_cap"`时权重是市值的平方根，`equal`时每只股票权重相同。行业因子收益受约束，按当期参与回归的股票市值加权后合计为0；只有一个行业时，该行业因子为0，市场水平留在国家因子里。风格列、行业集合或设计矩阵不满秩时，该期失败，不自动删因子。

因子协方差先用较长半衰期做相关、较短半衰期做波动，再用Bartlett核做滞后修正，年化后按近期已实现方差相对模型方差的比例缩放，最后做模拟的特征因子偏差修正：按当前矩阵模拟同样长度的收益，重新估计后比较每个特征向量的真实方差和估计方差，把平均比例乘回同一排序的特征值。`eigen_simulations=0`关闭这一步，`random_seed`让结果可复现，`eigen_scale`默认1.0。这些默认值不是供应商校准。特异风险用残差的指数加权方差；历史不足两期的股票使用对风格暴露做的对数线性预测，其余股票按观测数向该预测收缩。

当期行业必须在历史里有足够多的完整截面。某个行业刚出现、历史截面盖不住它时，估计失败并说明跳过了多少期，而不是把该行业因子收益记成0。`attribute_realized_return`用同一套回归把一天的已实现收益拆成因子贡献和特异收益。描述子混合用`blend_available_descriptors`：每只股票只对当期有值的成分重新归一权重，缺失成分不补造。成分要先标准化再混合，否则不同比率的单位会混在一起。

可运行示例见`examples/equity_style_risk.py`。风格描述子本身在`quant-factors`，包括多窗口换手、指数加权贝塔、短期反转、长期反转、季节性、分红、现金盈利、市场杠杆、资产负债率、收入增长、盈利波动、行业动量，以及调用方提供的分析师修正、预期盈利和预期增长。后三类没有输入时保持缺失。

## 组合风险、主动风险和序列化

`model.analyze(weights, benchmark_weights=None)`接收NAV分数权重。未持有资产可以显式给
0且无需模型覆盖；任何非零组合或基准权重都必须有X、D和Σ覆盖。返回对象包含：

- `portfolio_exposures=Xᵀw`；
- `benchmark_exposures=Xᵀw_b`和`active_exposures=Xᵀ(w-w_b)`；
- 组合年化方差和波动率；
- 主动组合年化跟踪误差方差和跟踪误差；
- 因子方差贡献`b⊙(Fb)`及资产特异方差贡献`w²⊙D`。

协方差、方差和方差贡献单位都是年化收益方差；波动率和跟踪误差是年化收益率；暴露
单位由调用方提供的X定义。因子协方差存在交叉项，因此单因子贡献可能为负，但贡献总和
与因子方差一致。

`model.to_dict()`、`model.diagnostics.to_dict()`和`report.to_dict()`均可直接写入JSON。
矩阵统一使用`index/columns/data`结构，避免DataFrame字典方向含混。诊断包含proxy标识、
窗口、每期截面数、秩、条件数、每个资产的特异收益样本数、当前X时间和全部收缩参数。

完整可运行示例见`examples/barra_style_proxy.py`。
