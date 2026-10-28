# BCRG-Net — code and derived experimental data

本仓库收录 BCRG-Net 的模型、训练、评估、HD95/ASSD 修正及确定性绘图代码，以及 Kvasir-SEG 实验的划分清单、汇总与逐图指标、归档预测。不含论文、成品图件、模型权重或原始图像与标注。

- 代码：`ESWA_KG_Polyp/src/`、`figure_revision_2026-10-28/figures/`。
- 实验数据：`ESWA_KG_Polyp/results/`，其中 `prediction_bundles/` 有 41 个运行预测包及索引；压缩包不含 GT 掩码。
- 指标修正：`ESWA_KG_Polyp/metric_correction/`。
- 每个发布文件的大小和 SHA-256：`RELEASE_MANIFEST.json`。

原始 Kvasir-SEG 图像和掩码请从[发布方](https://datasets.simula.no/kvasir-seg/)获取，并引用 [Jha 等的数据集论文](https://doi.org/10.1007/978-3-030-37734-2_37)。固定划分为 713/138/149 幅；运行 `python ESWA_KG_Polyp/src/verify_kvasir.py` 验证下载数据，运行 `python ESWA_KG_Polyp/src/bundle_predictions.py unpack` 解包预测。

此前使用的 `data/experiment_data_64x64.npz` 历史快照目前缺失。本仓库保留原始归档指标，不声称可逐字节重建该快照。`src/download_data.sh` 和 `src/verify_data.py` 含历史可选数据集路径；本文报告的实验只使用 Kvasir-SEG。
