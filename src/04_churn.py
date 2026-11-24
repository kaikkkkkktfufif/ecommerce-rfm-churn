"""Day 4：特征工程 + 流失预测模型。

输入：data/cleaned/uci_clean.csv
流失定义：用 2011-06-09 之前的数据算特征，2011-06-09 到 2011-09-07
这 90 天没购买的客户记为流失（特征和标签不重叠，不会作弊）。

特征（R、F、M 之外再加 7 个）：
- 平均购买间隔、购买间隔标准差、复购率、买过的品类数、
  退货率、客单价趋势（近期平均 / 早期平均）、周末购买占比

模型：随机森林 + XGBoost，train/test 按 8:2 分，
输出 ROC-AUC、F1、混淆矩阵，画特征重要性图。

输出：outputs/figures/ 下的特征重要性图和混淆矩阵
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CLEANED_DIR = BASE_DIR / "data" / "cleaned"
FIG_DIR = BASE_DIR / "outputs" / "figures"

FEATURE_END = "2011-06-09"   # 特征窗口截止
OBSERVE_END = "2011-09-07"   # 观察窗口截止（90 天后）


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # TODO: 1. 按特征窗口 / 观察窗口划分，生成流失标签
    # TODO: 2. 计算 R、F、M + 7 个新特征
    # TODO: 3. train/test 8:2 分，训练随机森林和 XGBoost
    # TODO: 4. 输出 ROC-AUC、F1、混淆矩阵，画特征重要性图


if __name__ == "__main__":
    main()
