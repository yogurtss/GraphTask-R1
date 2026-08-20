# GraphTask-R1 专利 PPT 素材包

本目录仅用于辅助制作专利介绍 PPT，不是正式权利要求书，也不构成专利性、自由实施或可授权性意见。

## 建议使用顺序

1. `01_四部分专利材料.md`：第一部分拆为 Technical Field、Prior Art、Problems of Prior Art 三页；第二部分包含整体 Summary 和四个核心发明点，之后为 Industrial Applications 与 Detectability。
2. `figures/fig1_related_work_comparison.*`：Prompt-based、Tool-based 与本项目路线的对比。
3. `figures/fig2_curriculum_v3_architecture_gpt_image.png`：GPT Image 生成的最新版整体架构图。
4. `figures/fig3_patent_method_flow.*`：黑白专利方法流程图草案。
5. `02_技术证据与待确认事项.md`：给发明人或专利代理人核对，不建议原样放入 PPT。
6. `03_参考资料.md`：相关论文与近似方向提醒。

## 图件说明

- 图 1、图 3：由 Python/Matplotlib 确定性生成，提供 SVG、PDF、PNG、TIFF 和源代码。
- 图 2：使用内置 GPT Image 生成；PNG 带透明通道，放到白色 PPT 背景上显示效果最佳。
- 图 2 的模块含义以 `01_四部分专利材料.md` 的“方法总览”与“主要步骤”为准；AI 图中文字仍应在提交前人工复核。

## 建议专利名称

**一种领域专用语言驱动的图智能体自进化训练方法**

## 一句话方案

让任务生成模型同时产生自然语言问题与受限类型化图程序，通过图执行器生成唯一 gold 并认证任务，再以接口就绪度门控的课程奖励、确定性归档和独立双适配器更新实现同轮闭环自演化。
