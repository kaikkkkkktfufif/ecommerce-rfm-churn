"""Day 2：RFM + K-Means 分群（UCI 主数据集）+ 多国家对比。

输入：data/cleaned/uci_clean.csv
- 以 2011-12-09 为基准日计算 R（距今多少天没买）、F（买了几次）、M（总共花了多少钱）
- 先取对数（log1p）再标准化，用肘部法 + 轮廓系数选 K，再 K-Means 分群
  （RFM 严重右偏，M 最大 58 万、中位数不到 900；不取对数的话 K-Means 会把几个
  超级大客户单独分成一群，只有 4 个人，没有业务意义）
- 输出每群的人数、占比、R/F/M 均值，画雷达图
- 英国、德国、法国、澳大利亚四国对比：各国 RFM 分布 + 各国客户在各群的占比
  （用全体客户训练的同一个模型给各国客户打标签，这样各国结果可以直接比；
  澳大利亚只有十几个客户，单独聚类没有意义）

输出：
- outputs/figures/ 下的图（肘部法、雷达图、四国 RFM 分布、四国客群占比）
- outputs/tables/ 下的结果表（CSV）
- data/cleaned/uci_rfm.csv：每个客户的 R/F/M 和所属群（Day 3 跨平台对比用）
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FIG_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

REF_DATE = "2011-12-09"
COUNTRIES = ["United Kingdom", "Germany", "France", "Australia"]
COUNTRY_CN = {"United Kingdom": "英国", "Germany": "德国", "France": "法国", "Australia": "澳大利亚"}
K_RANGE = range(2, 9)
SEED = 42

# 中文字体（Windows 自带微软雅黑）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def calc_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """每个客户一行：R、F、M 和所属国家。"""
    ref = pd.Timestamp(REF_DATE)
    rfm = df.groupby("Customer ID").agg(
        最后购买=("InvoiceDate", "max"),
        F=("Invoice", "nunique"),
        M=("Sales", "sum"),
    )
    rfm["R"] = (ref - rfm["最后购买"].dt.normalize()).dt.days
    # 少数客户在多个国家下过单，取下单次数最多的国家
    country = df.groupby(["Customer ID", "Country"])["Invoice"].nunique().reset_index()
    country = country.sort_values("Invoice", ascending=False).drop_duplicates("Customer ID")
    rfm["国家"] = country.set_index("Customer ID")["Country"]
    return rfm[["R", "F", "M", "国家"]]


def choose_k(X: np.ndarray) -> tuple[int, pd.DataFrame]:
    """肘部法 + 轮廓系数。K=2 只是把客户切成两半，对运营没用，所以在 K≥3 里取轮廓系数最高的。"""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        rows.append({"K": k, "SSE（肘部法）": km.inertia_, "轮廓系数": silhouette_score(X, km.labels_)})
    res = pd.DataFrame(rows)
    best_k = int(res.loc[res["K"] >= 3].sort_values("轮廓系数", ascending=False).iloc[0]["K"])

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(res["K"], res["SSE（肘部法）"], "o-", color="#1f77b4", label="SSE（肘部法）")
    ax1.set_xlabel("K（群数）")
    ax1.set_ylabel("SSE（群内误差平方和）", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(res["K"], res["轮廓系数"], "s--", color="#d62728", label="轮廓系数")
    ax2.set_ylabel("轮廓系数", color="#d62728")
    ax1.axvline(best_k, color="gray", linestyle=":")
    ax1.set_title(f"UCI 客户分群：肘部法 + 轮廓系数选 K（选定 K={best_k}）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_选K_肘部法_轮廓系数.png", dpi=150)
    plt.close(fig)
    return best_k, res


def name_clusters(profile: pd.DataFrame, overall: pd.Series) -> dict:
    """按各群 R/F/M 中位数和全体中位数比较，给每群起名字。R 越小越好，F、M 越大越好。
    用中位数不用均值：M 被几个超级大客户拉得很高，均值会让大部分群都显得“低于平均”。"""
    names = {}
    for c, row in profile.iterrows():
        recent = row["R中位数"] < overall["R"]
        valuable = (row["F中位数"] > overall["F"]) or (row["M中位数"] > overall["M"])
        if recent and valuable:
            names[c] = "高价值活跃客户"
        elif recent:
            names[c] = "新客户/潜力客户"
        elif valuable:
            names[c] = "流失风险客户"
        else:
            names[c] = "沉睡低价值客户"
    # 名字重复时按 M 均值区分
    for n in set(names.values()):
        same = [c for c in names if names[c] == n]
        if len(same) > 1:
            for i, c in enumerate(sorted(same, key=lambda c: -profile.loc[c, "M均值"])):
                names[c] = f"{n}{'ABCD'[i]}"
    return names


def radar(profile: pd.DataFrame) -> None:
    """雷达图：各群 R/F/M 均值（取对数后缩放到 0-1，R 反向，越靠外越好）。"""
    vals = pd.DataFrame({
        "最近购买\n（R 反向）": -np.log1p(profile["R均值"]),
        "购买次数 F": np.log1p(profile["F均值"]),
        "消费金额 M": np.log1p(profile["M均值"]),
    })
    vals = (vals - vals.min()) / (vals.max() - vals.min()) * 0.8 + 0.2
    labels = list(vals.columns)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for name, row in vals.iterrows():
        v = row.tolist() + [row.iloc[0]]
        ax.plot(angles, v, linewidth=2, label=f"{name}（{profile.loc[name, '人数']:,} 人）")
        ax.fill(angles, v, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("UCI 各客群 RFM 雷达图（越靠外越好）", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_客群雷达图.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def country_plots(rfm: pd.DataFrame, order: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """四国对比：RFM 分布箱线图 + 各群占比堆叠图。"""
    sub = rfm[rfm["国家"].isin(COUNTRIES)].copy()
    sub["国家中文"] = sub["国家"].map(COUNTRY_CN)
    cn_order = [COUNTRY_CN[c] for c in COUNTRIES]

    # 各国 RFM 中位数（中位数比均值更不受大客户影响）
    stat = sub.groupby("国家中文").agg(
        客户数=("R", "size"), R中位数=("R", "median"), F中位数=("F", "median"),
        M中位数=("M", "median"), M均值=("M", "mean"),
    ).loc[cn_order]

    # 箱线图（F、M 用对数坐标）
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, col, title, log in zip(
        axes, ["R", "F", "M"],
        ["R：距上次购买天数（越小越好）", "F：购买次数", "M：总消费金额（英镑）"], [False, True, True],
    ):
        data = [sub.loc[sub["国家中文"] == c, col] for c in cn_order]
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(cn_order) + 1))
        ax.set_xticklabels([f"{c}\n(n={len(d)})" for c, d in zip(cn_order, data)])
        ax.set_title(title)
        if log:
            ax.set_yscale("log")
    fig.suptitle("四国客户 RFM 分布对比（箱线图，不含离群点）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_四国RFM分布对比.png", dpi=150)
    plt.close(fig)

    # 各国客户在各群的占比
    share = pd.crosstab(sub["国家中文"], sub["客群"], normalize="index").loc[cn_order, order] * 100
    fig, ax = plt.subplots(figsize=(9, 4.8))
    left = np.zeros(len(share))
    for col in share.columns:
        ax.barh(share.index, share[col], left=left, label=col)
        for i, v in enumerate(share[col]):
            if v >= 6:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white")
        left += share[col].values
    ax.invert_yaxis()
    ax.set_xlabel("占该国客户的百分比（%）")
    ax.set_title("四国客户的客群构成对比")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(order), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_四国客群占比对比.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return stat, share


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    # 1. 读数据，算 R、F、M
    df = pd.read_csv(
        CLEANED_DIR / "uci_clean.csv",
        usecols=["Invoice", "InvoiceDate", "Customer ID", "Sales", "Country"],
        parse_dates=["InvoiceDate"],
    )
    rfm = calc_rfm(df)
    print(f"客户数：{len(rfm):,}（基准日 {REF_DATE}）")
    print(rfm[["R", "F", "M"]].describe().round(1).to_string())

    # 2. 取对数 + 标准化，选 K
    X = StandardScaler().fit_transform(np.log1p(rfm[["R", "F", "M"]]))
    best_k, k_res = choose_k(X)
    print("\n选 K 结果：")
    print(k_res.round(3).to_string(index=False))
    print(f"选定 K = {best_k}（K≥3 里轮廓系数最高）")
    k_res.round(4).to_csv(TABLE_DIR / "02_选K结果.csv", index=False, encoding="utf-8-sig")

    # 3. K-Means 分群 + 每群画像
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
    rfm["群号"] = km.labels_
    profile = rfm.groupby("群号").agg(
        人数=("R", "size"), R均值=("R", "mean"), F均值=("F", "mean"), M均值=("M", "mean"),
        R中位数=("R", "median"), F中位数=("F", "median"), M中位数=("M", "median"),
    )
    profile["占比%"] = profile["人数"] / len(rfm) * 100
    names = name_clusters(profile, rfm[["R", "F", "M"]].median())
    profile.index = profile.index.map(names)
    profile.index.name = "客群"
    profile = profile.sort_values("M均值", ascending=False)
    rfm["客群"] = rfm["群号"].map(names)
    order = list(profile.index)

    print("\n各客群画像：")
    print(profile.round(1).to_string())
    print(f"全体均值：R={rfm['R'].mean():.1f}  F={rfm['F'].mean():.1f}  M={rfm['M'].mean():.1f}")
    print(f"全体中位数：R={rfm['R'].median():.1f}  F={rfm['F'].median():.1f}  M={rfm['M'].median():.1f}")
    profile.round(2).to_csv(TABLE_DIR / "02_UCI客群画像.csv", encoding="utf-8-sig")
    radar(profile)

    # 4. 四国对比
    stat, share = country_plots(rfm, order)
    print("\n四国 RFM 对比：")
    print(stat.round(1).to_string())
    print("\n四国客群占比（%）：")
    print(share.round(1).to_string())
    stat.round(2).to_csv(TABLE_DIR / "02_四国RFM对比.csv", encoding="utf-8-sig")
    share.round(2).to_csv(TABLE_DIR / "02_四国客群占比.csv", encoding="utf-8-sig")

    # 5. 存客户级结果，Day 3 用
    rfm.drop(columns="群号").to_csv(CLEANED_DIR / "uci_rfm.csv", encoding="utf-8-sig")
    print(f"\n图已保存到 {FIG_DIR}")


if __name__ == "__main__":
    main()
