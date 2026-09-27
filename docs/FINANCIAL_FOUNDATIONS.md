# 10 穿透集中度风控

`quant_risk_monitor.lookthrough.check_fund_concentration(weights, disclosures, at, max_security_weight="0.1", max_unknown_weight="0")` 返回 allowed、breaches、coverage。默认未知暴露预算为零；security 超限报 LOOKTHROUGH_CONCENTRATION，未知超限报 LOOKTHROUGH_UNKNOWN。现金按 CASH:<currency> 身份处理。

这是只读风险检查 API，需调用方把结果纳入下单闸门；不会从模块导入自动修改旧风控规则。披露持仓是滞后证据，不等于实时头寸；用户容忍未知的阈值调整必须显式记录。tests/test_lookthrough.py 验证集中度及缺失披露阻断。
