"""Day 3：Olist 巴西电商同样流程 + 跨平台对比（英国 UCI vs 巴西 Olist）。

输入：
- data/cleaned/olist_clean.csv：Olist 每个用户一行（01_clean.py 生成）
- data/cleaned/olist_orders_clean.csv：Olist 每个订单一行
- data/cleaned/uci_clean.csv、data/cleaned/uci_rfm.csv：UCI 明细和 Day 2 的 RFM 结果

步骤：
1. Olist RFM：基准日取 Olist 数据最后一天（2018-08-29），R=距今天数，F=订单数，M=总金额（商品+运费，雷亚尔）
2. 和 UCI 一样：log1p + 标准化，肘部法 + 轮廓系数（K≥3 里取最高）选 K，K-Means 分群，按中位数给群命名
3. 巴西各州对比：用户最多的 4 个州（对应 UCI 的四国对比）
4. 跨平台对比：购买频率、客单价、复购率、首末次购买间隔，以及两边客群构成

注意：两个平台币种不同（英镑 vs 雷亚尔），金额只在各自平台内比较，跨平台只比不带单位的指标
（次数、比例、天数），或者用“相对本平台中位数的倍数”。

输出：outputs/figures/03_*.png、outputs/tables/03_*.csv、data/cleaned/olist_rfm.csv
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

K_RANGE = range(2, 9)
SEED = 42
STATE_CN = {"SP": "圣保罗州", "RJ": "里约州", "MG": "米纳斯州", "RS": "南里奥格兰德州"}

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def choose_k(X: np.ndarray) -> tuple[int, pd.DataFrame]:
    """和 Day 2 同样的规则：K≥3 里轮廓系数最高。轮廓系数用 1 万人抽样计算（9 万人全算太慢）。"""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
        sil = silhouette_score(X, km.labels_, sample_size=10000, random_state=SEED)
        rows.append({"K": k, "SSE（肘部法）": km.inertia_, "轮廓系数": sil})
    res = pd.DataFrame(rows)
    best_k = int(res.loc[res["K"] >= 3].sort_values("轮廓系数", ascending=False).iloc[0]["K"])

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(res["K"], res["SSE（肘部法）"], "o-", color="#1f77b4")
    ax1.set_xlabel("K（群数）")
    ax1.set_ylabel("SSE（群内误差平方和）", color="#1f77b4")
    ax2 = ax1.twinx()
    ax2.plot(res["K"], res["轮廓系数"], "s--", color="#d62728")
    ax2.set_ylabel("轮廓系数", color="#d62728")
    ax1.axvline(best_k, color="gray", linestyle=":")
    ax1.set_title(f"Olist 用户分群：肘部法 + 轮廓系数选 K（选定 K={best_k}）")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_Olist选K_肘部法_轮廓系数.png", dpi=150)
    plt.close(fig)
    return best_k, res


def name_clusters(profile: pd.DataFrame, overall: pd.Series) -> dict:
    """给群起名字。Olist 97% 用户只买过 1 次，F 几乎是常数，所以不能照搬 UCI 的命名：
    先看是否复购（F 均值 > 1.5），再看是否最近买过（R 中位数 < 全体中位数）；
    消费金额（M 中位数）明显高于全体（> 1.5 倍）才加“高消费”。"""
    names = {}
    for c, row in profile.iterrows():
        if row["F均值"] > 1.5:
            base = "复购用户"
        elif row["R中位数"] < overall["R"]:
            base = "近期一次性用户"
        else:
            base = "沉睡一次性用户"
        if row["M中位数"] > overall["M"] * 1.5:
            base = "高消费" + base
        names[c] = base
    # 名字重复时按 M 中位数区分
    for n in set(names.values()):
        same = [c for c in names if names[c] == n]
        if len(same) > 1:
            for i, c in enumerate(sorted(same, key=lambda c: -profile.loc[c, "M中位数"])):
                names[c] = f"{n}{'ABCD'[i]}"
    return names


def radar(profile: pd.DataFrame) -> None:
    vals = pd.DataFrame({
        "最近购买\n（R 反向）": -np.log1p(profile["R均值"]),
        "购买次数 F": np.log1p(profile["F均值"]),
        "消费金额 M": np.log1p(profile["M均值"]),
    })
    vals = (vals - vals.min()) / (vals.max() - vals.min()) * 0.8 + 0.2
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    for name, row in vals.iterrows():
        v = row.tolist() + [row.iloc[0]]
        ax.plot(angles, v, linewidth=2, label=f"{name}（{profile.loc[name, '人数']:,} 人）")
        ax.fill(angles, v, alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(vals.columns), fontsize=11)
    ax.set_yticklabels([])
    ax.set_title("Olist 各用户群 RFM 雷达图（越靠外越好）", pad=20, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_Olist客群雷达图.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def platform_metrics(name: str, orders: pd.DataFrame, cust_col: str, date_col: str, amt_col: str) -> dict:
    """orders：每个订单一行。返回不带币种的跨平台指标 + 本币客单价。"""
    per_cust = orders.groupby(cust_col).agg(
        订单数=(amt_col, "size"), 首次=(date_col, "min"), 末次=(date_col, "max"), 总额=(amt_col, "sum")
    )
    repeat = per_cust[per_cust["订单数"] > 1]
    first_gap = (
        orders.sort_values(date_col).groupby(cust_col)[date_col]
        .apply(lambda s: (s.iloc[1] - s.iloc[0]).days if len(s) > 1 else np.nan).dropna()
    )
    span_days = (orders[date_col].max() - orders[date_col].min()).days
    return {
        "平台": name,
        "数据跨度（天）": span_days,
        "用户数": len(per_cust),
        "订单数": len(orders),
        "人均订单数": per_cust["订单数"].mean(),
        "订单数中位数": per_cust["订单数"].median(),
        "复购率%（买过 ≥2 次）": len(repeat) / len(per_cust) * 100,
        "买过 ≥5 次的用户%": (per_cust["订单数"] >= 5).mean() * 100,
        "复购用户首两单间隔中位数（天）": first_gap.median(),
        "客单价均值（本币）": orders[amt_col].mean(),
        "客单价中位数（本币）": orders[amt_col].median(),
        "客单价 P90 / 中位数（倍）": orders[amt_col].quantile(0.9) / orders[amt_col].median(),
        "前 10% 用户贡献销售额%": per_cust["总额"].sort_values(ascending=False)
        .head(max(1, len(per_cust) // 10)).sum() / per_cust["总额"].sum() * 100,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    # ---------- 1. Olist RFM ----------
    users = pd.read_csv(CLEANED_DIR / "olist_clean.csv", parse_dates=["首次购买", "末次购买"])
    orders = pd.read_csv(CLEANED_DIR / "olist_orders_clean.csv", parse_dates=["order_purchase_timestamp"])
    ref = orders["order_purchase_timestamp"].max().normalize()
    rfm = pd.DataFrame({
        "R": (ref - users["末次购买"].dt.normalize()).dt.days.values,
        "F": users["订单数"].values,
        "M": users["总金额"].values,
        "州": users["州"].values,
    }, index=users["customer_unique_id"])
    print(f"Olist 用户数：{len(rfm):,}（基准日 {ref:%Y-%m-%d}，Olist 最后一天）")
    print(rfm[["R", "F", "M"]].describe().round(1).to_string())

    # ---------- 2. 选 K + 分群 ----------
    X = StandardScaler().fit_transform(np.log1p(rfm[["R", "F", "M"]]))
    best_k, k_res = choose_k(X)
    print("\n选 K 结果：")
    print(k_res.round(3).to_string(index=False))
    print(f"选定 K = {best_k}（K≥3 里轮廓系数最高）")
    k_res.round(4).to_csv(TABLE_DIR / "03_Olist选K结果.csv", index=False, encoding="utf-8-sig")

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
    print("\nOlist 各用户群画像（金额单位：雷亚尔）：")
    print(profile.round(1).to_string())
    print(f"全体中位数：R={rfm['R'].median():.0f}  F={rfm['F'].median():.0f}  M={rfm['M'].median():.1f}")
    profile.round(2).to_csv(TABLE_DIR / "03_Olist客群画像.csv", encoding="utf-8-sig")
    radar(profile)

    # ---------- 3. 巴西四州对比 ----------
    states = list(STATE_CN)
    sub = rfm[rfm["州"].isin(states)].copy()
    sub["州名"] = sub["州"].map(lambda s: f"{STATE_CN[s]}（{s}）")
    st_order = [f"{STATE_CN[s]}（{s}）" for s in states]
    st_stat = sub.groupby("州名").agg(
        用户数=("R", "size"), R中位数=("R", "median"), 复购率=("F", lambda f: (f > 1).mean() * 100),
        M中位数=("M", "median"), M均值=("M", "mean"),
    ).loc[st_order]
    st_share = pd.crosstab(sub["州名"], sub["客群"], normalize="index").loc[st_order, order] * 100
    print("\n巴西四州对比：")
    print(st_stat.round(1).to_string())
    print("\n四州客群占比（%）：")
    print(st_share.round(1).to_string())
    st_stat.round(2).to_csv(TABLE_DIR / "03_巴西四州对比.csv", encoding="utf-8-sig")
    st_share.round(2).to_csv(TABLE_DIR / "03_巴西四州客群占比.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    left = np.zeros(len(st_share))
    for col in st_share.columns:
        ax.barh(st_share.index, st_share[col], left=left, label=col)
        for i, v in enumerate(st_share[col]):
            if v >= 6:
                ax.text(left[i] + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white")
        left += st_share[col].values
    ax.invert_yaxis()
    ax.set_xlabel("占该州用户的百分比（%）")
    ax.set_title("巴西用户最多的四个州：客群构成对比")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(order), fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_巴西四州客群占比.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- 4. 跨平台对比 ----------
    uci = pd.read_csv(
        CLEANED_DIR / "uci_clean.csv", usecols=["Invoice", "InvoiceDate", "Customer ID", "Sales"],
        parse_dates=["InvoiceDate"],
    )
    uci_orders = uci.groupby("Invoice").agg(
        **{"Customer ID": ("Customer ID", "first"), "InvoiceDate": ("InvoiceDate", "min"), "金额": ("Sales", "sum")}
    )
    m_uci = platform_metrics("英国 UCI", uci_orders, "Customer ID", "InvoiceDate", "金额")
    m_bra = platform_metrics("巴西 Olist", orders, "customer_unique_id", "order_purchase_timestamp", "订单金额")
    comp = pd.DataFrame([m_uci, m_bra]).set_index("平台").T
    print("\n跨平台用户画像对比（客单价单位：英国=英镑，巴西=雷亚尔）：")
    print(comp.round(2).to_string())
    comp.round(3).to_csv(TABLE_DIR / "03_跨平台对比.csv", encoding="utf-8-sig")

    # 订单数分布：1 / 2 / 3-4 / 5-9 / 10+
    bins, labels = [0, 1, 2, 4, 9, np.inf], ["1 次", "2 次", "3–4 次", "5–9 次", "10 次以上"]
    uci_f = pd.read_csv(CLEANED_DIR / "uci_rfm.csv")["F"]
    dist = pd.DataFrame({
        "英国 UCI": pd.cut(uci_f, bins, labels=labels).value_counts(normalize=True).reindex(labels) * 100,
        "巴西 Olist": pd.cut(rfm["F"], bins, labels=labels).value_counts(normalize=True).reindex(labels) * 100,
    })
    print("\n购买次数分布（% 用户）：")
    print(dist.round(1).to_string())
    dist.round(2).to_csv(TABLE_DIR / "03_跨平台购买次数分布.csv", encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    # 图 a：购买次数分布
    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, dist["英国 UCI"], 0.4, label="英国 UCI", color="#4c72b0")
    axes[0].bar(x + 0.2, dist["巴西 Olist"], 0.4, label="巴西 Olist", color="#55a868")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("占用户百分比（%）")
    axes[0].set_title("购买次数分布")
    axes[0].legend()
    # 图 b：关键比例
    keys = ["复购率%（买过 ≥2 次）", "买过 ≥5 次的用户%", "前 10% 用户贡献销售额%"]
    short = ["复购率", "买过≥5次", "前10%用户\n贡献销售额"]
    xb = np.arange(len(keys))
    for off, (col, color) in zip([-0.2, 0.2], [("英国 UCI", "#4c72b0"), ("巴西 Olist", "#55a868")]):
        vals = comp.loc[keys, col].astype(float).values
        axes[1].bar(xb + off, vals, 0.4, label=col, color=color)
        for xi, v in zip(xb + off, vals):
            axes[1].text(xi, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels(short)
    axes[1].set_ylabel("%")
    axes[1].set_title("复购与销售集中度")
    axes[1].legend()
    # 图 c：客单价分布（各自除以本平台中位数，去掉币种影响）
    rel_uci = uci_orders["金额"] / uci_orders["金额"].median()
    rel_bra = orders["订单金额"] / orders["订单金额"].median()
    axes[2].boxplot([rel_uci, rel_bra], showfliers=False)
    axes[2].set_xticks([1, 2])
    axes[2].set_xticklabels([
        f"英国 UCI\n（中位数 £{uci_orders['金额'].median():.0f}）",
        f"巴西 Olist\n（中位数 R${orders['订单金额'].median():.0f}）",
    ])
    axes[2].axhline(1, color="gray", linestyle=":")
    axes[2].set_ylabel("客单价 / 本平台中位数（倍）")
    axes[2].set_title("客单价分布（相对本平台中位数）")
    fig.suptitle("跨平台用户画像对比：英国 UCI vs 巴西 Olist", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_跨平台用户画像对比.png", dpi=150)
    plt.close(fig)

    # 两边客群构成并排
    uci_seg = pd.read_csv(CLEANED_DIR / "uci_rfm.csv")["客群"].value_counts(normalize=True) * 100
    bra_seg = rfm["客群"].value_counts(normalize=True).reindex(order) * 100
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, s, title, color in [
        (axes[0], uci_seg, "英国 UCI 客群构成（K=4）", "#4c72b0"),
        (axes[1], bra_seg, f"巴西 Olist 客群构成（K={best_k}）", "#55a868"),
    ]:
        ax.barh(s.index, s.values, color=color)
        for i, v in enumerate(s.values):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, max(s.values) * 1.2)
        ax.set_xlabel("占用户百分比（%）")
        ax.set_title(title)
    fig.suptitle("两个平台的客群构成", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_跨平台客群构成.png", dpi=150)
    plt.close(fig)

    rfm.drop(columns="群号").to_csv(CLEANED_DIR / "olist_rfm.csv", encoding="utf-8-sig")
    print(f"\n图已保存到 {FIG_DIR}")


if __name__ == "__main__":
    main()
