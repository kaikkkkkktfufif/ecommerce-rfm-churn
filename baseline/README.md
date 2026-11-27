# 基线项目（原版，未修改）

**这个文件夹里的 notebook 不是我们写的**，是我们找来当起点的开源半成品，原样保留，方便对比。

- 原作者：alifialistu
- 原项目：https://github.com/alifialistu/RFMAnalysis-KMeans
- 版本：commit `13b784f89837304a466ec0ae9978454833c137df`（2020-10-30）
- 许可证：Apache License 2.0，见同目录的 `LICENSE`
- 文件：`RFMAnalysis-OnlineRetail.ipynb`（原版，一个字没改）
- 没有放进来的：原项目附带的数据压缩包 `OnlineRetail-Dataset.zip`。它是旧版数据，我们用的是 Online Retail II。

## 它做了什么

旧版 Online Retail 数据 → 清洗 → R、F、M **各自单独**做 K-Means（各 K=3）→ 相加得 RFM 总分 → 按 M 分会员等级。

## 我们在哪里改进

- 复现它的方法，并和我们的方法对比：`src/02b_baseline_compare.py`
- 对比结果和改进点：`docs/参考项目.md`
- 我们自己的完整流程：`src/01_clean.py` 到 `src/05_timeseries.py`
