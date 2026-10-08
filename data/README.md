# Synthetic data only

本项目所有数据均为合成数据，仅用于技术路线验证和工程架构展示，不包含任何真实企业数据、车型数据或商业机密。

`synthetic/example_design.json` 是公开的合成参数示例。`processed/` 保存运行生成的
PyTorch 图数据，默认不进入 Git。一个数据文件含 `train`、`val`、`test` 和 `ood`
四组独立设计；OOD 测试厚度范围与训练范围严格分开。数据文件附带配置、种子、
单位和来源标记。仅加载本项目自己生成、可信的 `.pt` 文件。
