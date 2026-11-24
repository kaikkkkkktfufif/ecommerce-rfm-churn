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
输出：data/cleaned/uci_clean.csv

清洗规则（Olist）：
- 只保留已交付订单（order_status == 'delivered'）
- 按 customer_unique_id 汇总：订单数、总金额、首次/末次购买日期
输出：data/cleaned/olist_clean.csv
"""

import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEANED_DIR = BASE_DIR / "data" / "cleaned"

UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
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
    urllib.request.urlretrieve(url, dest)
    print(f"已保存：{dest}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 下载 UCI
    download(UCI_URL, RAW_DIR / "online_retail_II.xlsx")
    # 2. 下载 Olist 三张表
    for name in OLIST_FILES:
        download(OLIST_BASE + name, RAW_DIR / name)

    # TODO: 3. 清洗 UCI，存 data/cleaned/uci_clean.csv
    # TODO: 4. 清洗 Olist，存 data/cleaned/olist_clean.csv
    # TODO: 5. 打印两个数据集的行数、时间范围、UCI 的国家分布前 10 名


if __name__ == "__main__":
    main()
