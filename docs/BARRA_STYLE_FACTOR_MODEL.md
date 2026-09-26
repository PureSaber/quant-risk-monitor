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

这是日频或其他等间隔收益的静态最小实现。它没有自动做市值加权回归、行业约束、
Newey-West调整、波动率regime缩放、特异风险贝叶斯分组或供应商式因子组合；如需这些
能力，应在有数据证据和独立预测验证后扩展。

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
