# 因子风险模型深化：暴露、估计与校准

本实现是一套可审计的公开统计风险模型，不复制供应商专有描述子或校准值。原有
`FactorModelConfig(model_kind=...)`仍使用等权 OLS、样本协方差和固定收缩；新增能力
需要显式配置。所有实验、参数选择和报告都是离线研究，不更改交易授权或风险限额。

## 1. 可复现的暴露生产

`ExposureBuildConfig`与`build_exposure_snapshot`接收两份独立的历史表：

- 描述子表：`asset/effective_at/available_at/source/market_cap/<descriptor columns>`。
  每行是一个资产的完整描述子快照，不能把部分字段更新当作完整快照。
- 分类表：`asset/effective_at/available_at/source/industry`。行业按历史生效时间和
  当时可得时间选择，不能用今天的分类重写过去。

每个时点、每个资产分别选择同时满足`effective_at <= as_of`、`available_at <= as_of`
的最新版本；同一生效时间以最新已知修订为准。拒绝重复版本、缺少来源、缺资产、
非正市值、无穷值、未知行业和配置所声明的过期数据。未来版本的描述子值不参与处理。
资产池、行业全集由调用方预先声明；本实现不认证股票池是否包含退市股票。

生产顺序固定如下：

1. 根据`styles={style: {descriptor: signed_weight}}`选择描述子。权重按绝对值之和
   归一化；负权重用于明确反向定义，零权重不要求输入。
2. 缺失值默认拒绝。可明确选择`industry_median`，仅使用同一历史时点同一行业的
   原始非缺失值中位数，不让已填补值继续成为其他资产的填补来源。默认至少三个有效
   同行业观测、每个描述子最多20%缺失；小组不足默认拒绝，可明确选择全截面回退。
3. 去极值可选`mad`（中位数 ± k×原始MAD）、`quantile`（两侧q分位）或`none`。
   默认k=5；MAD没有隐含1.4826系数。零MAD但非恒定的离散截面拒绝，需改用明确的
   分位数或不去极值规则。真实恒定描述子因无截面信息也拒绝。
4. 用明确的等权、市值或市值平方根权重计算加权均值与总体标准差，标准化每个描述子。
5. 加权组合描述子，再标准化组合。可声明`orthogonalize`依赖图，以带截距的加权
   投影去除指定风格影响，再标准化残差。图必须无环，顺序由依赖关系确定；JSON键顺序
   改变不会改变结果。因子暴露单位是该规则下的标准差单位。
6. 根据声明加入常数`market`和全部`industry:<name>`列。保存当时的市值、分类、
   因子列表、所选来源版本、去极值资产、填补资产/来源、均值、标准差和正交化系数。

`public_equity_style_config(industry_universe)`提供与quant-factors公开描述子目录匹配的
十类研究预设：规模、非线性规模、Beta、动量、残差波动、流动性、账面市值比、盈利收益率、
增长和杠杆。它是单描述子的公开近似方案，非供应商十风格复制；可替换为明确的多描述子组合。
预设将非线性规模对规模正交化，残差波动对Beta与规模正交化，最低截面30个资产。
描述子的财报口径、币种和市场收益基准仍须由上游数据/因子生产者保证。

## 2. 加权回归和行业识别

回归目标为`min sum_i w_i (r_i - X_i f)^2`。`regression_weighting`支持`equal`、
`market_cap`、`sqrt_market_cap`，后两者必须使用对应期初暴露快照里的正市值。
实现对设计矩阵和收益乘`sqrt(w)`；市值平方根权重不是再次把回归权重平方。

同时包含市场列和全部行业列时，设置`industry_constraint=true`。约束为
`sum_g C_g f_g = 0`，其中`C_g`为该期估计资产池中的行业市值占比。行业列允许非负
分数暴露，但逐资产必须合计1；每个声明行业必须有正市值覆盖。使用约束矩阵的正交
零空间求解，保留全部行业，不任意删除一个行业，也不靠ridge掩盖共线性。

在加权、施加约束后的自由设计空间检查秩和条件数；资产数必须超过自由参数数。
输出逐期的自由参数数、条件数、行业约束权重、约束误差和加权残差平方和。
行业约束随历史市值变化，使用当期暴露；后来的市值或分类不能反向改变历史回归。

## 3. 协方差、期限和波动状态

`factor_half_life`、`specific_half_life`用观测期数定义指数衰减半衰期；`null`表示等权。
时间权重归一化为a。中心化均值为`sum(a*r)`，零阶协方差分母为`1-sum(a²)`，
所以等权且滞后为0时与`ddof=1`样本协方差相同。报告有效样本数`1/sum(a²)`；
过短半衰期导致有效样本数退化时拒绝。

Newey–West 使用Bartlett核。令`u_t=sqrt(a_t)*(r_t-weighted_mean)`，则

```
F_rate = [sum_t u_t u_t' + sum(l=1..L) (1-l/(L+1))
          * sum_t (u_t u_(t-l)' + u_(t-l) u_t')] / (1-sum(a²))
```

该加权构造是PSD Bartlett核的合同变换。最后按配置向对角阵收缩，只有数值舍入级负
特征值可以裁零，且记录修复；显著非PSD或非有限输出会失败。年化只乘一次A。

**期限不能混用。** `risk_horizon=H`明确风险期限，`newey_west_lags`和
`specific_newey_west_lags`分别只能为0或H−1，并且小于训练样本数。H=1不能启用
正滞后。非零滞后的矩阵是H期线性收益和方差除以H的年化风险率，H期预测方差为
`H/A * w'Σw`；它不是单日潜在方差。滞后为0的H期预测作无自相关的线性缩放。
多期验证须提供匹配期限的独立`outcomes`，请求的`horizon_periods`必须等于H；
收益聚合采用线性近似，不能把它宣称为精确复利账户损益。

`regime_window=0`关闭状态调整。启用时，对最近完整窗口中的每一期，只用它之前的
历史均值与零阶方差形成标准化创新，计算截面平均平方创新，再按`regime_half_life`
汇总成共同方差乘数。因子风险与特异风险分别估计乘数；需至少
`regime_min_periods + regime_window`个观测。零方差历史对应非零创新会失败。
`regime_scale_min/max`是显式的方差乘数界限，报告原值、应用值和是否触界。
这是透明的先验预测误差调整方案，不是MSCI的专有VRA实现或市场校准参数。

## 4. 分组与贝叶斯特异风险

基础特异方差支持独立半衰期和匹配期限的Newey–West。`specific_target=global`
保留总体均值目标；`industry`使用当前历史分类，同业其他资产的平均特异方差，
不把资产自身放入自己的行业目标。默认至少三个组内资产，薄组默认拒绝；显式设置
`specific_group_fallback=global`才回退到其他资产全体，报告所有回退资产。

`specific_shrinkage_method=fixed`保留配置的固定收缩。
`bayesian`使用逆Gamma先验均值收缩形式：

`posterior_variance = (df * sample_variance + k * peer_target) / (df + k)`，

其中`df=effective_observations−1`、`k=specific_prior_dof`。样本可靠性降低时，
同样先验强度对应更高收缩。指数加权与HAC下这是明确的有效自由度近似，不能宣称严格
后验覆盖率。按行业的目标、实际收缩权重、有效样本数及状态调整均输出诊断。
高级时间衰减、HAC或状态调整要求资产有连续完整残差历史，避免把停牌/缺期压缩成相邻期。

## 5. 校准与证据复核

新增实际命令：

```sh
python examples/advanced_barra_research.py runs/advanced-inputs
quant-risk build-exposures --input runs/advanced-inputs/exposures.json --out runs/exposures
quant-risk verify-research --run runs/exposures
quant-risk calibrate-risk --input runs/advanced-inputs/calibration.json --out runs/calibration
quant-risk verify-research --run runs/calibration
```

所有输出目录必须不存在。示例明确为12个合成资产、36个合成收益期、三个行业、两个
风格；同时使用加权/行业约束、EWMA、HAC、状态调整与行业贝叶斯收缩。示例没有真实
市场预测准确性或收益结论。

校准输入`quant-risk.calibration-study/v1`固定2—32组完整候选配置、基线、回看长度、
频率、历史暴露、收益、选择期请求、留出期请求、`plan_available_at`、
`selection_as_of`和`evaluation_as_of`。候选计划必须在首个选择预测之前声明；
选择期全部收益必须在冻结时已可得，留出期在冻结之后才开始。
所有候选在相同组合、相同日期和相同期限下，以选择期组合QLIKE排序，平分按候选ID。
任何候选失败或QLIKE因零风险而不可用时，整个研究失败，不静默删掉差的候选。
随后只评价选中模型和预先指定的基线，不根据留出期结果重新选参数。

结果保存每个候选选择期逐期预测证据、分数、选择SHA、选中参数及两组留出期指标。
改变留出期收益不会改变已冻结的选择SHA。这里没有自动提升为生产配置的动作。
时间戳、来源分类和候选预登记由生产者声明；SHA只能绑定内容，不能认证作者或证明
历史登记发生过。`market_calibration_certified`始终为false。

`verify-research`检查固定文件集合、文件类型与哈希，并从原始输入完全重算输出；
修改结果再重写清单哈希仍会被拒绝。原始资料和结果同时全部替换不能仅凭自签清单识别，
正式前向研究仍需外部不可变登记。复核历史产物应使用产物生成时冻结的源码与依赖版本。

## 6. 研究配方接入

quant-lab的`risk_model.exposure_pipeline`声明构建配置、描述子字段到历史字段的映射、
市值字段和行业字段；`risk_model.estimation`声明高级估计设置。A股适配器读取QDK原始
PIT历史逐字段选择，保留各自时间与历史记录，不以当天读取时间刷新数据年龄。
构建后的Σ送入现有优化器，绝对/主动暴露、行业限额和跟踪误差继续通过已有目标检查。
报告明确`risk_horizon`和对应期限方差。

## 方法与验证来源

- [Statsmodels HAC文档](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)：Newey–West/Bartlett的公开统计背景；本实现直接估计因子收益协方差，并不调用回归系数标准误接口。
- [MSCI公开的回归权重说明](https://www.msci.com/research-and-insights/blog-post/equity-factor-commentary-a-magnificent-7-factor)：平方根市值权重的背景。这里只使用公开数学定义，不声称复制MSCI产品。

独立测试通过KKT方程核对受约束WLS，以双重求和核对加权Bartlett协方差，以手算后验
均值核对组内收缩，检查未来数据扰动、原始历史年龄、正交化、期限年化、组合风险恒等式、
不可篡改重算及留出期隔离。功能测试通过不等于真实市场校准通过。
