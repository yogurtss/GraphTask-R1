# Translation and source notes

- 本附件按用户指定范围，仅处理训练数据、验证集和最终测试集相关段落，不是全文双语翻译。
- 读取版本为 arXiv:2608.01904v1。PDF 为 9 页；arXiv 页面显示“10 pages”与下载文件页数存在差异，可能源自元数据或版式计数。
- `paper.md` 中 Original 为忠实压缩后的英文证据块，不是逐字复制整段；数字、数据集名和实验协议保持不变。
- 正式训练 source manifest、正式 chain pool 和 seed/validation sample IDs 未随仓库发布，相关字段无法从论文还原。
- 作者代码包含多种 dataset loader，但不能据此断言正式实验全部使用这些数据集。
- `assets/` 当前为空：本次报告不依赖图像或表格裁剪；所有数字均可由正文、主表或公开测试 JSONL 复核。
