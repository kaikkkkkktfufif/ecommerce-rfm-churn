"""Day 3：Olist 巴西电商同样流程 + 跨平台对比。

输入：data/cleaned/olist_clean.csv
- 对巴西电商数据做和 UCI 一样的 RFM + K-Means 分群（肘部法 + 轮廓系数选 K）
- 英国（UCI）vs 巴西（Olist）用户画像对比：购买频率、客单价、复购情况

输出：outputs/figures/ 下的对比图
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FIG_DIR = BASE_DIR / "outputs" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: 1. 读 olist_clean.csv，算 R、F、M
    # TODO: 2. 肘部法 + 轮廓系数选 K，K-Means 分群
    # TODO: 3. 英国 vs 巴西用户画像对比图（购买频率、客单价、复购情况）


if __name__ == "__main__":
    main()
