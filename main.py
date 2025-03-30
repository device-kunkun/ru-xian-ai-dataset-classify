import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             mean_squared_error, r2_score)


plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False


plt.rcParams['font.size'] = 12

# ==================== 分类模型评估 ====================
# 加载乳腺癌数据集
cancer = load_breast_cancer()
X_clf, y_clf = cancer.data, cancer.target

# 数据分割
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42)

# 数据标准化
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)


def evaluate_classification(model, X_train, X_test, y_train, y_test, model_name):
    """评估分类模型并可视化结果"""
    # 训练和预测
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # 计算指标
    metrics = {
        "准确率": accuracy_score(y_test, y_pred),
        "精确率": precision_score(y_test, y_pred),
        "召回率": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    # 打印指标
    print(f"\n{model_name}评估结果：")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    # 绘制混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt="d", cmap="Blues",
                annot_kws={"size": 14})
    plt.title(f"{model_name}混淆矩阵")
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.show()

    # 绘制ROC曲线
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title(f'{model_name} ROC曲线')
        plt.legend(loc="lower right")
        plt.show()


# 评估不同复杂度的模型
print("=" * 40 + " 分类模型评估 " + "=" * 40)

# 1. 适当拟合的模型（逻辑回归）
evaluate_classification(LogisticRegression(max_iter=1000, C=1),
                        X_train_clf_scaled, X_test_clf_scaled,
                        y_train_clf, y_test_clf, "逻辑回归")

# 2. 欠拟合模型（深度过小的决策树）
evaluate_classification(DecisionTreeClassifier(max_depth=2),
                        X_train_clf_scaled, X_test_clf_scaled,
                        y_train_clf, y_test_clf, "欠拟合决策树")

# 3. 过拟合模型（深度过大的决策树）
evaluate_classification(DecisionTreeClassifier(max_depth=20),
                        X_train_clf, X_test_clf,  # 决策树不需要标准化
                        y_train_clf, y_test_clf, "过拟合决策树")

# ==================== 回归模型评估 ====================
# 加载糖尿病数据集
diabetes = load_diabetes()
X_reg, y_reg = diabetes.data, diabetes.target

# 数据分割
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)


def evaluate_regression(model, X_train, X_test, y_train, y_test, model_name):
    """评估回归模型性能"""
    # 训练和预测
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 计算指标
    metrics_train = {
        "MSE": mean_squared_error(y_train, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "R²": r2_score(y_train, y_pred_train)
    }

    metrics_test = {
        "MSE": mean_squared_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R²": r2_score(y_test, y_pred_test)
    }

    # 打印结果
    print(f"\n{model_name}评估结果：")
    print("训练集指标:")
    for name, value in metrics_train.items():
        print(f"{name}: {value:.4f}")

    print("\n测试集指标:")
    for name, value in metrics_test.items():
        print(f"{name}: {value:.4f}")


print("\n" + "=" * 40 + " 回归模型评估 " + "=" * 40)

# 1. 欠拟合模型（线性回归）
evaluate_regression(LinearRegression(),
                    X_train_reg, X_test_reg,
                    y_train_reg, y_test_reg, "线性回归")

# 2. 过拟合模型（多项式回归）
# 创建多项式回归管道
poly_reg = make_pipeline(
    PolynomialFeatures(degree=3),  # 使用3次多项式
    StandardScaler(),
    LinearRegression()
)

evaluate_regression(poly_reg,
                    X_train_reg, X_test_reg,
                    y_train_reg, y_test_reg, "三次多项式回归")