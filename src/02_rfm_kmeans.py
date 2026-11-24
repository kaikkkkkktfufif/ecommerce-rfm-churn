"""Day 2：RFM + K-Means 分群（UCI 主数据集）+ 多国家对比。

输入：data/cleaned/uci_clean.csv
- 以 2011-12-09 为基准日计算 R（距今多少天没买）、F（买了几次）、M（总共花了多少钱）
- 标准化后用肘部法 + 轮廓系数选 K，再 K-Means 分群
- 输出每群的人数、占比、R/F/M 均值，画雷达图
- 单独对英国、德国、法国、澳大利亚各算一份 RFM 分布和分群结果，做对比图

输出：outputs/figures/ 下的雷达图和各国对比图
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FIG_DIR = BASE_DIR / "outputs" / "figures"

REF_DATE = "2011-12-09"
COUNTRIES = ["United Kingdom", "Germany", "France", "Australia"]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: 1. 读 uci_clean.csv，算 R、F、M
    # TODO: 2. 标准化，肘部法 + 轮廓系数选 K
    # TODO: 3. K-Means 分群，输出每群人数、占比、R/F/M 均值
    # TODO: 4. 画雷达图
    # TODO: 5. 四个国家的 RFM 分布和分群对比图


if __name__ == "__main__":
    main()
