# 基于电商数据的用户行为分析与客群分群

数据科学导论课程课题。两个目标：RFM + K-Means 客群分群，用户流失预测。
中期答辩后按老师反馈调整：加了第二个数据集、多国家对比、新特征和时序模型。

## 数据集

1. **UCI Online Retail II**（英国电商，2009 年 12 月–2011 年 12 月，约 106 万行）
   https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
2. **Olist 巴西电商**（2016–2018，约 10 万订单）
   https://github.com/olist/work-at-olist-data

数据文件不进仓库（文件大，可重新生成），运行 `01_clean.py` 会自动下载和清洗。

## 方法

- 分群：RFM + K-Means（肘部法 + 轮廓系数定 K）
- 基线对比：复现 GitHub 项目 alifialistu/RFMAnalysis-KMeans（Apache-2.0）的方法并和我们的方法对比，见 `docs/参考项目.md`
- 流失预测：随机森林 / XGBoost（ROC-AUC、F1）
- 时间规律：季节分解 + Holt-Winters + ARIMA，黑五/圣诞季效应

## 文件结构

```
data/raw/          原始下载文件（不进仓库）
data/cleaned/      清洗后的 CSV（不进仓库）
src/               代码，按编号顺序运行
outputs/figures/   所有图
report/            报告
docs/              一周冲刺计划、数据说明、参考项目
baseline/          基线项目原版 notebook（别人写的，Apache-2.0，未修改）
```

## 运行方法

```bash
conda create -n ds python=3.13
conda activate ds
pip install -r requirements.txt
python src/01_clean.py
python src/02_rfm_kmeans.py
python src/02b_baseline_compare.py   # 基线对比，要先跑 02
python src/03_olist.py
python src/04_churn.py
python src/05_timeseries.py
```

## 进度记录

| 日期 | 计划 | 负责人 | 状态 |
|---|---|---|---|
| Day 1：11 月 25 日（周三） | 环境 + 数据下载 + 清洗 | 两人 | ✅ 完成 |
| Day 2：11 月 26 日（周四） | RFM + K-Means 分群，多国家对比 | 罗祺骞 | ✅ 完成 |
| Day 3：11 月 27 日（周五） | Olist 同样流程，跨平台对比 | 罗祺骞 | ⬜ 未开始 |
| Day 4：11 月 28 日（周六） | 特征工程 + 流失预测模型 | 游泓柯 | ⬜ 未开始 |
| Day 5：11 月 29 日（周日） | 时间序列：分解、节日效应、预测 | 游泓柯 | ⬜ 未开始 |
| Day 6：11 月 30 日（周一） | 整合 + 写报告 | 两人 | ⬜ 未开始 |
| Day 7：12 月 1 日（周二） | 检查 + 定稿 | 两人 | ⬜ 未开始 |
| 缓冲：12 月 2 日（周三） | 补漏 + 提交 | 两人 | ⬜ 未开始 |

## 分工

- 罗祺骞：分群主线（数据预处理、RFM、K-Means、调参）
- 游泓柯：预测主线（特征工程、流失预测模型、可视化与商业策略）
