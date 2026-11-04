# V1人工构造语义评测种子 v1-seed-1

120条设计用例，六类各20条，56条多轮、41条对抗。不是线上用户采样，也不是已经执行的模型评测。

cases.json为人工可审查数据；user_turns按同一会话逐轮执行，末轮目标由expected给出。fixture_requirements是待实现的业务真值规范，尚未转换成全部可执行fixture。A/B/C代表合成商品101/102/103；未特别指定时A199黑、B299白、C399黑。订单O1/O2/O3、身份U1/U2及政策P1等仅是合成语义标识，不能当未来商城DTO的真实字段类型或端点。

readiness.business_available只代表现有对应业务已有代码，不代表本用例必通过；runnable_now全部false，因为评测runner与各例fixture尚待实现。政策、订单、人工还分别受K-B、K-C/K-D、K-E阻塞，不能以“尚未接入”固定答复满足目标。全部execution_status=not_run。

allowed_tool_plan在现有业务中引用当前真实函数，steps是允许涉及的函数，constraints说明条件/顺序和调用上限，不能强制全调用。缺契约业务只写语义步骤，不造函数/HTTP接口。allowed_final_intents允许有意义的等价行为，存在争议需人工复核；金额、身份、商品ID等必须程序核对，不能只用关键词或LLM裁判。

目前所有用例都是development_seed。使用它们调Prompt后不能再声称是独立留出集；paraphrase_group用于未来按整组划分，分组须再次审核语义近似。发版前仍须遵守docs/08-测试/测试计划.md的至少200业务样本、50多轮、另40安全挑战、3次真实模型评测要求。本120条种子不降低该门槛。

从项目根目录运行数据自检：

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest tests.v1.i.test_seed_dataset
```

该命令仅解析JSON和读取源码AST校验函数引用，不导入Agent、不调用模型、不访问网络、不写SQLite。它不产生任何准确率或幻觉率成绩。

下一步：主代理核对争议标注→按真实schema实现fixture/runner→冻结Prompt/model/fixture hash与预算→运行模型→程序校验金额权限轨迹+人工语义rubric→按分类报告通过/失败/blocked。blocked不计pass；业务覆盖完成度另报，不能只缩小分母报高分。
