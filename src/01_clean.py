"""Day 1：数据下载与清洗。

下载：
- UCI Online Retail II（英国电商，2009-2011）：
  https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
- Olist 巴西电商（2016-2018，GitHub 镜像）：
  https://github.com/olist/work-at-olist-data

清洗规则（UCI）：
- 两个 sheet（Year 2009-2010、Year 2010-2011）合并
- 去掉 Customer ID 为空的行
- Quantity <= 0 的行：先按客户统计退货次数，再去掉
- Price <= 0 的去掉；去重
输出：data/cleaned/uci_clean.csv（交易明细）
      data/cleaned/uci_returns.csv（退货/取消明细，Day 4 算退货率用）

清洗规则（Olist）：
- 只保留已交付订单（order_status == 'delivered'）
- 按 customer_unique_id 汇总：订单数、总金额、首次/末次购买日期
输出：data/cleaned/olist_clean.csv（每个用户一行）
      data/cleaned/olist_orders_clean.csv（每个订单一行，Day 5 时序用）

最后自动生成 docs/数据说明.md（里面的数字全部来自本脚本运行结果）。
"""

import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
DOC_PATH = BASE_DIR / "docs" / "数据说明.md"

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_ZIP = RAW_DIR / "online_retail_ii.zip"
UCI_XLSX = RAW_DIR / "online_retail_II.xlsx"

OLIST_BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets/"
OLIST_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
]


def download(url: str, dest: Path) -> None:
    """下载文件，已存在就跳过。"""
    if dest.exists():
        print(f"已存在，跳过：{dest.name}")
        return
    print(f"下载中：{url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)  # 下完才改名，中途断了不会留下坏文件
    print(f"已保存：{dest}")


def get_uci_xlsx() -> None:
    """下载 UCI 压缩包并解压出 xlsx。"""
    download(UCI_URL, UCI_ZIP)
    if UCI_XLSX.exists():
        return
    with zipfile.ZipFile(UCI_ZIP) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".xlsx"))
        with zf.open(name) as src, open(UCI_XLSX, "wb") as dst:
            dst.write(src.read())
    print(f"已解压：{UCI_XLSX.name}")


def clean_uci() -> dict:
    """清洗 UCI，返回统计信息（写数据说明用）。"""
    print("读取 UCI 两个 sheet（文件大，要 1-3 分钟）……")
    sheets = pd.read_excel(UCI_XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    stats = {"sheets": {k: len(v) for k, v in sheets.items()}}
    df = pd.concat(sheets.values(), ignore_index=True)
    stats["raw_rows"] = len(df)

    # 1. 去掉 Customer ID 为空的行
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    stats["after_drop_null_id"] = len(df)

    # 2. Quantity <= 0 是退货/取消：先单独存下来（Day 4 按时间窗口算退货率），再去掉
    returns = df[df["Quantity"] <= 0].drop_duplicates()
    returns.to_csv(CLEANED_DIR / "uci_returns.csv", index=False)
    stats["return_rows"] = len(returns)
    stats["return_customers"] = returns["Customer ID"].nunique()
    df = df[df["Quantity"] > 0]

    # 3. Price <= 0 去掉
    stats["price_le0_rows"] = int((df["Price"] <= 0).sum())
    df = df[df["Price"] > 0]

    # 4. 去重（两个 sheet 在 2010 年 12 月有重叠，重复行也在这里去掉）
    before = len(df)
    df = df.drop_duplicates()
    stats["dup_rows"] = before - len(df)

    df["Sales"] = df["Quantity"] * df["Price"]  # 每行销售额
    df = df.sort_values("InvoiceDate")
    df.to_csv(CLEANED_DIR / "uci_clean.csv", index=False)

    # 每个客户的退货次数（按退货订单号计）
    ret_cnt = returns.groupby("Customer ID")["Invoice"].nunique()

    stats.update(
        clean_rows=len(df),
        customers=df["Customer ID"].nunique(),
        invoices=df["Invoice"].nunique(),
        date_min=df["InvoiceDate"].min(),
        date_max=df["InvoiceDate"].max(),
        n_countries=df["Country"].nunique(),
        total_sales=df["Sales"].sum(),
        top_countries=df.groupby("Country")["Customer ID"].nunique().sort_values(ascending=False).head(10),
        top_countries_rows=df["Country"].value_counts(),
        max_returns=int(ret_cnt.max()),
    )
    return stats


def clean_olist() -> dict:
    """清洗 Olist，返回统计信息。"""
    orders = pd.read_csv(RAW_DIR / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(RAW_DIR / "olist_order_items_dataset.csv")
    customers = pd.read_csv(RAW_DIR / "olist_customers_dataset.csv")
    stats = {"raw_orders": len(orders), "status_counts": orders["order_status"].value_counts()}

    # 1. 只保留已交付订单
    orders = orders[orders["order_status"] == "delivered"]

    # 2. 订单金额 = 该订单所有商品价格 + 运费
    amount = items.groupby("order_id").agg(
        商品金额=("price", "sum"), 运费=("freight_value", "sum"), 商品件数=("order_item_id", "count")
    )
    amount["订单金额"] = amount["商品金额"] + amount["运费"]

    # 3. 用 customer_unique_id 认人（customer_id 是一单一号）
    od = (
        orders.merge(customers[["customer_id", "customer_unique_id", "customer_state"]], on="customer_id")
        .merge(amount, left_on="order_id", right_index=True)
    )
    stats["orders_no_items"] = len(orders) - len(od)
    od = od[["order_id", "customer_unique_id", "customer_state", "order_purchase_timestamp",
             "商品件数", "商品金额", "运费", "订单金额"]].sort_values("order_purchase_timestamp")
    od.to_csv(CLEANED_DIR / "olist_orders_clean.csv", index=False)

    # 4. 按用户汇总
    user = od.groupby("customer_unique_id").agg(
        订单数=("order_id", "nunique"),
        总金额=("订单金额", "sum"),
        首次购买=("order_purchase_timestamp", "min"),
        末次购买=("order_purchase_timestamp", "max"),
        州=("customer_state", "first"),
    ).reset_index()
    user.to_csv(CLEANED_DIR / "olist_clean.csv", index=False)

    stats.update(
        delivered_orders=len(od),
        users=len(user),
        repeat_users=int((user["订单数"] > 1).sum()),
        date_min=od["order_purchase_timestamp"].min(),
        date_max=od["order_purchase_timestamp"].max(),
        total_sales=od["订单金额"].sum(),
        top_states=user["州"].value_counts().head(5),
    )
    return stats


def write_doc(u: dict, o: dict) -> None:
    """把统计结果写成数据说明（Markdown）。"""
    d = lambda t: t.strftime("%Y-%m-%d")
    lines = [
        "# 数据说明（由 src/01_clean.py 自动生成）",
        "",
        "## 数据集一：UCI Online Retail II（英国电商）",
        "",
        f"- 原始行数：{u['raw_rows']:,}（" + "、".join(f"{k} {v:,} 行" for k, v in u["sheets"].items()) + "）",
        f"- 去掉 Customer ID 为空后：{u['after_drop_null_id']:,} 行",
        f"- 退货/取消行（Quantity ≤ 0）：{u['return_rows']:,} 行，涉及 {u['return_customers']:,} 个客户，"
        "另存为 `uci_returns.csv`",
        f"- 去掉 Price ≤ 0：{u['price_le0_rows']:,} 行；去掉重复行：{u['dup_rows']:,} 行",
        f"- **清洗后：{u['clean_rows']:,} 行，{u['customers']:,} 个客户，{u['invoices']:,} 个订单**",
        f"- 时间范围：{d(u['date_min'])} 至 {d(u['date_max'])}",
        f"- 国家数：{u['n_countries']}；总销售额：£{u['total_sales']:,.0f}",
        "",
        "国家分布前 10（按客户数）：",
        "",
        "| 国家 | 客户数 | 交易行数 |",
        "|---|---|---|",
    ]
    for c, n in u["top_countries"].items():
        lines.append(f"| {c} | {n:,} | {u['top_countries_rows'].get(c, 0):,} |")
    lines += [
        "",
        "## 数据集二：Olist 巴西电商",
        "",
        f"- 原始订单数：{o['raw_orders']:,}；其中已交付（delivered）：{o['status_counts']['delivered']:,}",
        f"- 已交付但没有商品明细的订单：{o['orders_no_items']:,}（已去掉）",
        f"- **清洗后：{o['delivered_orders']:,} 个订单，{o['users']:,} 个用户（按 customer_unique_id）**",
        f"- 买过 2 次及以上的用户：{o['repeat_users']:,}（占 {o['repeat_users'] / o['users']:.1%}）",
        f"- 时间范围：{d(o['date_min'])} 至 {d(o['date_max'])}",
        f"- 总销售额（商品 + 运费）：R${o['total_sales']:,.0f}",
        "",
        "用户最多的 5 个州：" + "、".join(f"{s}（{n:,}）" for s, n in o["top_states"].items()),
        "",
        "## 输出文件（data/cleaned/，不进仓库）",
        "",
        "- `uci_clean.csv`：UCI 交易明细，多一列 Sales = Quantity × Price",
        "- `uci_returns.csv`：UCI 退货/取消明细",
        "- `olist_clean.csv`：Olist 每个用户一行（订单数、总金额、首次/末次购买、州）",
        "- `olist_orders_clean.csv`：Olist 每个订单一行（金额、日期，时序分析用）",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成：{DOC_PATH}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 下载
    get_uci_xlsx()
    for name in OLIST_FILES:
        download(OLIST_BASE + name, RAW_DIR / name)

    # 2. 清洗
    u = clean_uci()
    o = clean_olist()

    # 3. 打印结果
    print("\n===== UCI =====")
    print(f"清洗后 {u['clean_rows']:,} 行，{u['customers']:,} 个客户，"
          f"{u['date_min']:%Y-%m-%d} 至 {u['date_max']:%Y-%m-%d}")
    print("国家分布前 10（客户数）：")
    print(u["top_countries"].to_string())
    print("\n===== Olist =====")
    print(f"清洗后 {o['delivered_orders']:,} 个订单，{o['users']:,} 个用户，"
          f"{o['date_min']:%Y-%m-%d} 至 {o['date_max']:%Y-%m-%d}")

    write_doc(u, o)


if __name__ == "__main__":
    main()
