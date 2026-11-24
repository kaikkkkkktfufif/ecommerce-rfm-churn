"""Day 5：时间序列。

输入：data/cleaned/uci_clean.csv
- 按天汇总销售额，用 statsmodels 做季节分解（趋势 + 季节 + 残差）
- 节日效应：黑五（2010-11-26、2011-11-25）和圣诞节前两周的日均销售额，
  跟全年日均比，输出高出百分之几
- 时序模型：Holt-Winters + ARIMA（statsmodels），预测未来 30 天，对比 MAPE
- 预警规则：预测销量超过过去 30 天均值 1.5 倍时，建议提前备货

输出：outputs/figures/ 下的分解图、节日对比图、预测图
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FIG_DIR = BASE_DIR / "outputs" / "figures"

BLACK_FRIDAY = ["2010-11-26", "2011-11-25"]
ALERT_MULTIPLIER = 1.5  # 预测销量超过过去 30 天均值的倍数就预警


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: 1. 按天汇总销售额
    # TODO: 2. 季节分解，画趋势、季节、残差图
    # TODO: 3. 黑五和圣诞季 vs 全年日均对比
    # TODO: 4. Holt-Winters + ARIMA 预测未来 30 天，对比 MAPE，画预测图


if __name__ == "__main__":
    main()
