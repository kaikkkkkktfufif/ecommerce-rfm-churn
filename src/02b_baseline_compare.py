"""Day 2 补充：基线项目复现 + 对比实验。

基线项目：alifialistu/RFMAnalysis-KMeans（Apache-2.0）
https://github.com/alifialistu/RFMAnalysis-KMeans
（参考版本 commit 13b784f，2020-10-30，详见 docs/参考项目.md）

基线的做法（按它的 notebook 原样复现，只把数据换成我们清洗后的 Online Retail II）：
- R = 距数据最后一笔交易的天数；F = 交易**行数**（不是订单数）；M = 总金额
- R、F、M **各自单独**做 MinMax 缩放 + K-Means（K=3），不取对数
- 三个维度的群号按好坏排序（0 最差、2 最好）后相加，得到 RFM 总分（0–6）
- 会员等级只看 M 的群：Bronze / Silver / Gold

我们的做法（02_rfm_kmeans.py）：F 用订单数；R、F、M 先 log1p 再标准化，三维一起聚类；
肘部法 + 轮廓系数选 K（=4）。

对比指标（同一个评价空间：log1p + 标准化后的 R/F/M 三维）：
- 轮廓系数、Calinski-Harabasz 指数、Davies-Bouldin 指数
- 各群人数是否均衡（最小群占比）

输出：outputs/figures/02b_基线vs改进_*.png、outputs/tables/02b_基线vs改进.csv
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"
SEED = 42

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def baseline_segments(df: pd.DataFrame) -> pd.DataFrame:
    """按基线 notebook 的方法复现。"""
    maxdate = df["InvoiceDate"].max()
    rfm = pd.DataFrame({
        "Recency": (maxdate - df.groupby("Customer ID")["InvoiceDate"].max()).dt.days,
        "Frequency": df.groupby("Customer ID")["Invoice"].count(),  # 基线用交易行数
        "Monetary": df.groupby("Customer ID")["Sales"].sum(),
    })
    for col, good_is_high in [("Recency", False), ("Frequency", True), ("Monetary", True)]:
        x = MinMaxScaler().fit_transform(rfm[[col]])
        lab = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=SEED).fit_predict(x)
        # 和基线的 cluster_order 一样：按均值排序，0 最差、2 最好
        means = pd.Series(rfm[col].values).groupby(lab).mean().sort_values(ascending=good_is_high)
        rank = {c: i for i, c in enumerate(means.index)}
        rfm[col + "Cluster"] = [rank[c] for c in lab]
    rfm["RFMScore"] = rfm[["RecencyCluster", "FrequencyCluster", "MonetaryCluster"]].sum(axis=1)
    rfm["Membership"] = np.select(
        [rfm["MonetaryCluster"] > 1, rfm["MonetaryCluster"] > 0], ["Gold", "Silver"], "Bronze"
    )
    return rfm


def evaluate(X: np.ndarray, labels, name: str) -> dict:
    labels = np.asarray(labels)
    sizes = pd.Series(labels).value_counts()
    return {
        "方案": name,
        "群数": len(sizes),
        "轮廓系数（越大越好）": silhouette_score(X, labels),
        "CH 指数（越大越好）": calinski_harabasz_score(X, labels),
        "DB 指数（越小越好）": davies_bouldin_score(X, labels),
        "最小群人数": int(sizes.min()),
        "最小群占比%": sizes.min() / len(labels) * 100,
        "最大群占比%": sizes.max() / len(labels) * 100,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    df = pd.read_csv(
        CLEANED_DIR / "uci_clean.csv",
        usecols=["Invoice", "InvoiceDate", "Customer ID", "Sales"],
        parse_dates=["InvoiceDate"],
    )
    base = baseline_segments(df)
    ours = pd.read_csv(CLEANED_DIR / "uci_rfm.csv", index_col="Customer ID")  # 02_rfm_kmeans.py 的结果
    base = base.loc[ours.index]

    # 统一评价空间：我们的 R/F/M 取 log1p 后标准化
    X = StandardScaler().fit_transform(np.log1p(ours[["R", "F", "M"]]))

    # 基线单维度聚类的问题：看各维度 3 个群的人数
    print("基线：R/F/M 各自聚成 3 群后的人数（0 最差 → 2 最好）")
    dim_sizes = pd.DataFrame({
        c: base[c + "Cluster"].value_counts().sort_index() for c in ["Recency", "Frequency", "Monetary"]
    }).fillna(0).astype(int)
    print(dim_sizes.to_string())
    print("\n基线会员等级人数：")
    print(base["Membership"].value_counts().to_string())

    rows = [
        evaluate(X, base["RFMScore"], "基线：RFM 总分（0–6 分组）"),
        evaluate(X, base["Membership"], "基线：会员等级（只看 M）"),
        evaluate(X, ours["客群"], "改进：log + 三维 K-Means（K=4）"),
    ]
    res = pd.DataFrame(rows).set_index("方案")
    print("\n基线 vs 改进（同一评价空间：log1p + 标准化的 R/F/M）：")
    print(res.round(3).to_string())
    res.round(4).to_csv(TABLE_DIR / "02b_基线vs改进.csv", encoding="utf-8-sig")
    dim_sizes.to_csv(TABLE_DIR / "02b_基线各维度群人数.csv", encoding="utf-8-sig")

    # 图 1：三种分群的各群人数占比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    groups = [
        ("基线：RFM 总分", base["RFMScore"].value_counts(normalize=True).sort_index() * 100),
        ("基线：会员等级", base["Membership"].value_counts(normalize=True).reindex(["Bronze", "Silver", "Gold"]) * 100),
        ("改进：K-Means 四群", ours["客群"].value_counts(normalize=True) * 100),
    ]
    for ax, (title, s) in zip(axes, groups):
        ax.bar([str(i) for i in s.index], s.values, color="#4c72b0")
        for i, v in enumerate(s.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
        ax.set_title(title)
        ax.set_ylabel("占客户百分比（%）")
        ax.set_ylim(0, max(s.values) * 1.15)
        ax.tick_params(axis="x", labelsize=9, rotation=15)
    fig.suptitle("基线 vs 改进：各群人数分布", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02b_基线vs改进_群人数分布.png", dpi=150)
    plt.close(fig)

    # 图 2：三个指标对比
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    short = ["基线\nRFM总分", "基线\n会员等级", "改进\nK-Means"]
    for ax, col in zip(axes, ["轮廓系数（越大越好）", "CH 指数（越大越好）", "DB 指数（越小越好）"]):
        bars = ax.bar(short, res[col].values, color=["#999999", "#bbbbbb", "#dd8452"])
        for b, v in zip(bars, res[col].values):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
        ax.set_title(col)
    fig.suptitle("基线 vs 改进：聚类质量指标（同一评价空间）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02b_基线vs改进_聚类指标.png", dpi=150)
    plt.close(fig)
    print(f"\n图已保存到 {FIG_DIR}")


if __name__ == "__main__":
    main()
