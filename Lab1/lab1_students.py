import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 100

RANDOM_STATE = 42
TEST_SIZE = 0.2
FILE_NAME = "StudentsPerformance.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=100, bbox_inches="tight")
    plt.show()
    plt.close()


def header(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


# ======================================================================
header("1. ЗАГРУЗКА ДАННЫХ")
# ======================================================================
if not os.path.exists(FILE_NAME):
    raise FileNotFoundError(f"Файл '{FILE_NAME}' не найден в {os.getcwd()}")

df = pd.read_csv(FILE_NAME)
print(f"Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
# короткие имена столбцов
df.columns = ["gender", "race", "parent_edu", "lunch", "prep", "math", "reading", "writing"]
print(df.head(10))

SCORES = ["math", "reading", "writing"]
CATS = ["gender", "race", "parent_edu", "lunch", "prep"]


# ======================================================================
header("2. ПЕРВИЧНЫЙ АНАЛИЗ ДАННЫХ")
# ======================================================================
print("\n--- Типы и пропуски ---")
print(pd.DataFrame({"тип": df.dtypes.astype(str), "пропусков": df.isnull().sum(),
                    "уникальных": df.nunique()}))
print(f"\nПолных дубликатов строк: {df.duplicated().sum()}")

print("\n--- Описательная статистика ---")
print(df.describe().round(2))
for col in SCORES:
    print(f"Асимметрия {col}: {stats.skew(df[col]):.3f}")

# Распределение баллов (целевая переменная — math)
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for ax, col, color in zip(axes, SCORES, ["#4C72B0", "#55A868", "#C44E52"]):
    ax.hist(df[col], bins=30, color=color)
    ax.axvline(df[col].mean(), color="black", ls="--", label=f"среднее = {df[col].mean():.1f}")
    ax.axvline(df[col].median(), color="orange", ls="-.", label=f"медиана = {df[col].median():.0f}")
    ax.set_title(f"{col} score" + (" (целевая переменная)" if col == "math" else ""))
    ax.set_xlabel("Балл")
    ax.set_ylabel("Число учеников")
    ax.legend()
save("01_scores")

# Категориальные признаки
fig, axes = plt.subplots(1, 5, figsize=(24, 4.8))
for ax, col in zip(axes, CATS):
    vc = df[col].value_counts()
    ax.bar(vc.index, vc.values, color="#8172B2")
    ax.bar_label(ax.containers[0])
    ax.set_title(col)
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylabel("Число учеников")
save("02_categorical")

# Связь баллов между собой
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (xcol, ycol) in zip(axes, [("reading", "math"), ("writing", "math"), ("reading", "writing")]):
    sns.scatterplot(data=df, x=xcol, y=ycol, hue="gender", hue_order=["female", "male"],
                    alpha=0.5, s=20, ax=ax)
    ax.set_title(f"{ycol} vs {xcol} (r = {df[xcol].corr(df[ycol]):.3f})")
save("03_scatter")

# Балл по математике по группам
print("\n--- Средние баллы по группам ---")
for col in CATS:
    print(df.groupby(col)[SCORES].mean().round(1), "\n")
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
for ax, col in zip(axes, CATS):
    order = df.groupby(col)["math"].median().sort_values().index
    sns.boxplot(data=df, x=col, y="math", order=order, ax=ax)
    ax.tick_params(axis="x", rotation=35)
    ax.set_title(f"math — {col}")
save("04_groups")

# Корреляции исходных признаков (бинарные закодированы 0/1)
enc = df.assign(gender_male=(df["gender"] == "male").astype(int),
                lunch_standard=(df["lunch"] == "standard").astype(int),
                prep_none=(df["prep"] == "none").astype(int))
corr_raw = enc[["math", "reading", "writing", "gender_male", "lunch_standard", "prep_none"]].corr()
print("--- Корреляции исходных признаков ---")
print(corr_raw.round(3))
plt.figure(figsize=(7.5, 6))
sns.heatmap(corr_raw, annot=True, fmt=".3f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
plt.title("Корреляционная матрица исходных признаков")
save("05_corr_raw")


# ======================================================================
header("3. ПРЕДОБРАБОТКА ДАННЫХ")
# ======================================================================
print(f"Пропусков всего: {df.isnull().sum().sum()}, дубликатов: {df.duplicated().sum()}")

print("\n--- Выбросы по правилу 1.5·IQR ---")
for col in SCORES:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    out = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
    print(f"  {col:8s}: границы [{q1 - 1.5 * iqr:.2f}; {q3 + 1.5 * iqr:.2f}], выбросов {len(out)}: "
          f"{sorted(out[col].tolist())}")

q1, q3 = df["math"].quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("\nУдаляемые наблюдения (выбросы целевой переменной):")
print(df[(df["math"] < low) | (df["math"] > high)])

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
sns.boxplot(data=df[SCORES], orient="h", ax=axes[0])
axes[0].set_title(f"До удаления выбросов (n = {len(df)})")
df = df[(df["math"] >= low) & (df["math"] <= high)].reset_index(drop=True)
sns.boxplot(data=df[SCORES], orient="h", ax=axes[1])
axes[1].set_title(f"После удаления выбросов math (n = {len(df)})")
for ax in axes:
    ax.set_xlabel("Балл")
save("06_outliers")
print(f"\nОсталось строк: {len(df)}")

# Кодирование категориальных переменных (One-Hot)
data = pd.get_dummies(df, columns=CATS, drop_first=True, dtype=int)
data = data.rename(columns={
    "race_group B": "race_B", "race_group C": "race_C", "race_group D": "race_D", "race_group E": "race_E",
    "parent_edu_bachelor's degree": "parent_bachelor", "parent_edu_high school": "parent_high_school",
    "parent_edu_master's degree": "parent_master", "parent_edu_some college": "parent_some_college",
    "parent_edu_some high school": "parent_some_high_school",
})
X = data.drop(columns="math")
y = data["math"]
print(f"\nМатрица признаков: {X.shape}, признаки: {list(X.columns)}")

# Разделение 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")
print(f"Средний балл math: train = {y_train.mean():.2f}, test = {y_test.mean():.2f}")

# Стандартизация (параметры только по train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ======================================================================
header("4. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ И VIF")
# ======================================================================
corr = pd.concat([X, y], axis=1).corr()
plt.figure(figsize=(13, 11))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 8}, square=True, cbar_kws={"shrink": 0.75})
plt.title("Матрица корреляций признаков и целевой переменной (math)")
save("07_corr_full")

print("--- Корреляция признаков с math ---")
print(corr["math"].drop("math").sort_values(ascending=False).round(3))
pairs = (corr.drop(index="math", columns="math")
         .where(np.triu(np.ones((X.shape[1], X.shape[1]), dtype=bool), k=1)).stack())
print("\n--- Пары признаков с |r| > 0.5 ---")
print(pairs[pairs.abs() > 0.5].sort_values(key=abs, ascending=False).round(3))


def calc_vif(frame):
    Xc = sm.add_constant(frame.astype(float))
    return pd.DataFrame({
        "Признак": frame.columns,
        "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(frame.shape[1])],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)


vif_data = calc_vif(X)
print("\n--- VIF ---")
print(vif_data.round(2).to_string(index=False))
print("\nVIF без writing:")
print(calc_vif(X.drop(columns="writing")).head(3).round(2).to_string(index=False))

plt.figure(figsize=(9, 6.5))
colors = ["#C44E52" if v >= 10 else "#DD8452" if v >= 5 else "#55A868" for v in vif_data["VIF"]]
plt.barh(vif_data["Признак"], vif_data["VIF"], color=colors)
plt.axvline(5, ls="--", c="gray", label="VIF = 5 (умеренная)")
plt.axvline(10, ls="--", c="black", label="VIF = 10 (сильная)")
plt.gca().invert_yaxis()
plt.xlabel("VIF")
plt.title("VIF-коэффициенты признаков")
plt.legend()
save("08_vif")


# ======================================================================
header("5. РЕГРЕССИОННЫЕ МОДЕЛИ НА ИСХОДНЫХ ПРИЗНАКАХ")
# ======================================================================
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
ALPHAS_LASSO = np.logspace(-3, 1, 30)
ALPHAS_RIDGE = np.logspace(-3, 3, 30)


def build_models(pca_components=None):
    """Пайплайны: стандартизация -> (PCA) -> модель; alpha подбирается по 5-fold CV."""
    def steps(est):
        s = [("scaler", StandardScaler())]
        if pca_components is not None:
            s.append(("pca", PCA(n_components=pca_components, random_state=RANDOM_STATE)))
        s.append(("model", est))
        return Pipeline(s)

    return {
        "Linear": steps(LinearRegression()),
        "Lasso": GridSearchCV(steps(Lasso(max_iter=100_000)), {"model__alpha": ALPHAS_LASSO},
                              cv=cv, scoring="neg_root_mean_squared_error"),
        "Ridge": GridSearchCV(steps(Ridge()), {"model__alpha": ALPHAS_RIDGE},
                              cv=cv, scoring="neg_root_mean_squared_error"),
    }


def evaluate(models, label):
    rows, preds = [], {}
    for name, model in models.items():
        cv_res = cross_validate(model, X_train, y_train, cv=cv,
                                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        rows.append({
            "Признаки": label, "Модель": name,
            "alpha": model.best_params_["model__alpha"] if hasattr(model, "best_params_") else np.nan,
            "CV RMSE": -cv_res["test_rmse"].mean(), "CV RMSE std": cv_res["test_rmse"].std(),
            "CV R²": cv_res["test_r2"].mean(),
            "Test RMSE": np.sqrt(mean_squared_error(y_test, p)),
            "Test R²": r2_score(y_test, p),
            "Test MAPE, %": mean_absolute_percentage_error(y_test, p) * 100,
        })
    return pd.DataFrame(rows), preds


models_orig = build_models()
res_orig, preds_orig = evaluate(models_orig, f"Исходные ({X.shape[1]})")
print(res_orig.round(4).to_string(index=False))

# Подбор alpha
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, name, alphas in [(axes[0], "Lasso", ALPHAS_LASSO), (axes[1], "Ridge", ALPHAS_RIDGE)]:
    gs = models_orig[name]
    mean = -gs.cv_results_["mean_test_score"]
    std = gs.cv_results_["std_test_score"]
    ax.semilogx(alphas, mean, marker="o", ms=3)
    ax.fill_between(alphas, mean - std, mean + std, alpha=0.2)
    ax.axvline(gs.best_params_["model__alpha"], c="red", ls="--",
               label=f"лучшее α = {gs.best_params_['model__alpha']:.4g}")
    ax.set_xlabel("α (логарифмическая шкала)")
    ax.set_ylabel("RMSE на кросс-валидации, баллы")
    ax.set_title(f"{name}: подбор параметра регуляризации")
    ax.legend()
save("09_alpha")

# Коэффициенты
coefs = pd.DataFrame({
    "Linear": models_orig["Linear"].named_steps["model"].coef_,
    "Lasso": models_orig["Lasso"].best_estimator_.named_steps["model"].coef_,
    "Ridge": models_orig["Ridge"].best_estimator_.named_steps["model"].coef_,
}, index=X.columns).sort_values("Linear")
print("\n--- Коэффициенты (на стандартизованных признаках), баллы ---")
print(coefs.round(3))
print(f"Lasso обнулило: {list(coefs.index[coefs['Lasso'].abs() < 1e-8])}")
raw_lr = LinearRegression().fit(X_train, y_train)
print("\nКоэффициенты в исходных единицах (Linear):")
print(pd.Series(raw_lr.coef_, index=X.columns).round(3))
coefs.plot(kind="barh", figsize=(10, 7), width=0.8)
plt.axvline(0, c="black", lw=0.8)
plt.xlabel("Коэффициент при стандартизованном признаке, баллы")
plt.title("Коэффициенты моделей на исходных признаках")
save("10_coefficients")

# Прогноз vs факт
male_test = X_test["gender_male"].values == 1
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (name, p) in zip(axes, preds_orig.items()):
    ax.scatter(y_test[~male_test], p[~male_test], s=12, alpha=0.5, label="девушки")
    ax.scatter(y_test[male_test], p[male_test], s=12, alpha=0.5, label="юноши")
    ax.plot([20, 100], [20, 100], "k--", lw=1, label="идеальный прогноз")
    ax.set_xlabel("Фактический балл math")
    ax.set_ylabel("Прогноз")
    ax.set_title(f"{name}: прогноз vs факт")
    ax.legend()
save("11_pred_vs_actual")


# ======================================================================
header("6. АНАЛИЗ ОСТАТКОВ")
# ======================================================================
def residual_analysis(preds, label, fig_name):
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    rows = []
    for i, (name, p) in enumerate(preds.items()):
        resid = y_test.values - p
        ax = axes[0, i]
        ax.scatter(p, resid, s=12, alpha=0.5)
        ax.axhline(0, color="red", lw=1.5)
        order = np.argsort(p)
        roll = pd.Series(np.abs(resid[order])).rolling(30, center=True).mean()
        ax.plot(p[order], roll, color="orange", lw=2, label="скользящее среднее |остатка|")
        ax.set_title(f"{name}: остатки vs прогноз ({label})")
        ax.set_xlabel("Прогноз, баллы")
        ax.set_ylabel("Остаток, баллы")
        ax.legend(loc="upper right", fontsize=8)
        sns.histplot(resid, kde=True, ax=axes[1, i], color="#55A868", bins=25)
        axes[1, i].axvline(0, color="red")
        axes[1, i].set_title(f"{name}: распределение остатков")
        axes[1, i].set_xlabel("Остаток, баллы")
        _, bp_p, _, _ = het_breuschpagan(resid, sm.add_constant(p))
        _, sw_p = stats.shapiro(resid)
        rows.append({"Признаки": label, "Модель": name, "Среднее": resid.mean(),
                     "Std": resid.std(), "Асимметрия": stats.skew(resid),
                     "BP p-value": bp_p,
                     "Вывод BP": "гетероскедастичность" if bp_p < 0.05 else "гомоскедастичность",
                     "Shapiro p-value": sw_p})
    save(fig_name)
    return pd.DataFrame(rows)


print(residual_analysis(preds_orig, "исходные", "12_residuals_orig").round(4).to_string(index=False))


# ======================================================================
header("7. МЕТОД ГЛАВНЫХ КОМПОНЕНТ (PCA)")
# ======================================================================
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_scaled)
eigvals = pca_full.explained_variance_
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
VAR_THRESHOLD = 0.95
n_components = int(np.searchsorted(cumulative, VAR_THRESHOLD) + 1)
print(pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(eigvals))],
                    "λ": eigvals.round(3), "доля, %": (explained * 100).round(2),
                    "накопл., %": (cumulative * 100).round(2)}).to_string(index=False))
print(f"\nКритерий Кайзера (λ > 1): {(eigvals > 1).sum()} компонент")
print(f"Для 90% дисперсии нужно {int(np.searchsorted(cumulative, 0.90) + 1)} компонент")
print(f"Для {VAR_THRESHOLD:.0%} дисперсии нужно {n_components} компонент (из {X.shape[1]})")

k = np.arange(1, len(eigvals) + 1)
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(k, eigvals, "o-", color="#4C72B0")
axes[0].axhline(1, c="red", ls="--", label="критерий Кайзера (λ = 1)")
axes[0].axvline(n_components, c="green", ls="--", label=f"выбрано {n_components} компонент")
axes[0].set_xlabel("Номер главной компоненты")
axes[0].set_ylabel("Собственное значение λ")
axes[0].set_title("График каменистой осыпи (scree plot)")
axes[0].set_xticks(k)
axes[0].legend()
axes[1].bar(k, explained * 100, alpha=0.7, label="доля компоненты")
axes[1].step(k, cumulative * 100, where="mid", color="red", label="накопленная доля")
axes[1].axhline(VAR_THRESHOLD * 100, ls="--", c="gray", label=f"порог {VAR_THRESHOLD:.0%}")
axes[1].axvline(n_components, ls="--", c="green")
axes[1].set_xlabel("Номер главной компоненты")
axes[1].set_ylabel("Объяснённая дисперсия, %")
axes[1].set_title("Объяснённая и накопленная дисперсия")
axes[1].set_xticks(k)
axes[1].legend(loc="center right")
save("13_pca_scree")

loadings = pd.DataFrame(pca_full.components_.T, index=X.columns,
                        columns=[f"PC{i+1}" for i in range(len(eigvals))])
plt.figure(figsize=(14, 7.5))
sns.heatmap(loadings, cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 8})
plt.axvline(n_components, color="black", lw=3)
plt.title("Нагрузки главных компонент (справа от чёрной линии — отброшенные)")
save("14_pca_loadings")

pcs_train = pd.DataFrame(
    PCA(n_components=n_components, random_state=RANDOM_STATE).fit_transform(X_train_scaled),
    columns=[f"PC{i+1}" for i in range(n_components)])
vif_pca = calc_vif(pcs_train)
print("\n--- VIF главных компонент ---")
print(vif_pca.round(3).to_string(index=False))
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
sns.heatmap(pcs_train.corr(), cmap="coolwarm", center=0, vmin=-1, vmax=1, annot=True,
            fmt=".2f", annot_kws={"size": 7}, ax=axes[0])
axes[0].set_title("Корреляции между главными компонентами")
axes[1].bar([f"Исходные признаки\n({X.shape[1]})", f"Главные компоненты\n({n_components})"],
            [vif_data["VIF"].max(), vif_pca["VIF"].max()], color=["#C44E52", "#55A868"])
axes[1].bar_label(axes[1].containers[0], fmt="%.2f")
axes[1].axhline(10, ls="--", c="black", label="VIF = 10")
axes[1].set_ylabel("Максимальный VIF")
axes[1].set_title("Мультиколлинеарность до и после PCA")
axes[1].legend()
save("15_pca_vif")


# ======================================================================
header("8. РЕГРЕССИОННЫЕ МОДЕЛИ НА ГЛАВНЫХ КОМПОНЕНТАХ")
# ======================================================================
models_pca = build_models(pca_components=n_components)
res_pca, preds_pca = evaluate(models_pca, f"PCA ({n_components})")
print(res_pca.round(4).to_string(index=False))
print()
print(residual_analysis(preds_pca, "PCA", "16_residuals_pca").round(4).to_string(index=False))


# ======================================================================
header("9. СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
# ======================================================================
comparison = pd.concat([res_orig, res_pca], ignore_index=True)
print(comparison.round(4).to_string(index=False))
comparison.round(4).to_csv("metrics_comparison.csv", index=False, encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, metric in zip(axes, ["Test RMSE", "Test R²", "Test MAPE, %"]):
    sns.barplot(data=comparison, x="Модель", y=metric, hue="Признаки", ax=ax)
    ax.set_title(metric)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    ax.get_legend().remove()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Признаки", loc="lower center", ncol=2)
fig.suptitle("Сравнение метрик: исходные признаки vs главные компоненты")
fig.tight_layout(rect=(0, 0.1, 1, 1))
plt.savefig(os.path.join(FIG_DIR, "17_comparison.png"), dpi=100, bbox_inches="tight")
plt.show()
plt.close()

# Качество в зависимости от числа компонент
r2_by_k = []
for k_ in range(1, X.shape[1] + 1):
    m = Pipeline([("scaler", StandardScaler()), ("pca", PCA(n_components=k_)),
                  ("model", LinearRegression())])
    r2_by_k.append(cross_validate(m, X_train, y_train, cv=cv, scoring="r2")["test_score"].mean())
print("\n--- CV R² линейной регрессии в зависимости от числа компонент ---")
print(pd.DataFrame({"k": k, "CV R²": np.round(r2_by_k, 4),
                    "накопл. дисперсия": cumulative.round(4)}).to_string(index=False))
plt.figure(figsize=(11, 5))
plt.plot(k, r2_by_k, "o-", label="CV R² линейной регрессии на k компонентах")
plt.plot(k, cumulative, "s--", c="gray", label="накопленная доля дисперсии X")
plt.axvline(n_components, c="green", ls="--", label=f"выбрано {n_components} компонент")
plt.xlabel("Число главных компонент k")
plt.ylabel("Значение")
plt.xticks(k)
plt.title("Качество модели и объяснённая дисперсия признаков в зависимости от числа компонент")
plt.legend()
save("18_r2_by_k")

# Прогноз vs факт: исходные признаки и PCA
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, (label, preds) in zip(axes, [("исходные признаки", preds_orig),
                                     (f"PCA, {n_components} компонент", preds_pca)]):
    p = preds["Linear"]
    ax.scatter(y_test[~male_test], p[~male_test], s=12, alpha=0.5, label="девушки")
    ax.scatter(y_test[male_test], p[male_test], s=12, alpha=0.5, label="юноши")
    ax.plot([20, 100], [20, 100], "k--", lw=1)
    ax.set_xlabel("Фактический балл math")
    ax.set_ylabel("Прогноз")
    ax.set_title(f"Linear: прогноз vs факт ({label})")
    ax.legend()
save("19_pred_compare")

best = comparison.loc[comparison["Test RMSE"].idxmin()]
print(f"\nЛучшая модель: {best['Модель']} на признаках «{best['Признаки']}» "
      f"(RMSE = {best['Test RMSE']:.3f}, R² = {best['Test R²']:.4f})")


# ======================================================================
header("10. АНАЛИЗ ПОГРЕШНОСТИ И ПОВЫШЕНИЕ ТОЧНОСТИ")
# ======================================================================
p_best = preds_orig["Linear"]
resid = y_test.values - p_best
abs_err = np.abs(resid)
print(f"Доля учеников с ошибкой ≤ 5 баллов: {(abs_err <= 5).mean():.1%}")
print(f"Доля учеников с ошибкой > 10 баллов: {(abs_err > 10).mean():.1%}")
print(f"Максимальная ошибка: {abs_err.max():.1f} балла")

# Ошибка в зависимости от уровня балла
bins = pd.cut(y_test, [26, 50, 70, 100], labels=["27–50", "51–70", "71–100"])
err_by_level = pd.DataFrame({"уровень": bins.values, "y": y_test.values, "p": p_best}).groupby("уровень").apply(
    lambda g: pd.Series({"учеников": len(g),
                         "RMSE": np.sqrt(mean_squared_error(g["y"], g["p"])),
                         "MAPE, %": mean_absolute_percentage_error(g["y"], g["p"]) * 100,
                         "средний остаток": (g["y"] - g["p"]).mean()}))
print("\n--- Ошибка в зависимости от уровня балла ---")
print(err_by_level.round(2))

# Ошибка в разрезе групп
print("\n--- Средний остаток и его разброс по группам ---")
groups = {"gender_male": ("девушки", "юноши"), "lunch_standard": ("льготное", "обычное"),
          "prep_none": ("курс пройден", "без курса")}
rows = []
for col, (n0, n1) in groups.items():
    for val, nm in [(0, n0), (1, n1)]:
        m = X_test[col].values == val
        rows.append({"признак": col, "группа": nm, "учеников": m.sum(),
                     "средний остаток": resid[m].mean(), "std остатка": resid[m].std()})
err_by_group = pd.DataFrame(rows)
print(err_by_group.round(2).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.boxplot(x=bins.values, y=resid, ax=axes[0], color="#8DA0CB")
axes[0].axhline(0, color="red")
axes[0].set_xlabel("Фактический балл math")
axes[0].set_ylabel("Остаток, баллы")
axes[0].set_title("Остатки в зависимости от уровня балла")
eg = err_by_group.assign(label=err_by_group["группа"])
axes[1].barh(eg["label"], eg["средний остаток"], xerr=eg["std остатка"], color="#66C2A5",
             error_kw={"alpha": 0.4})
axes[1].axvline(0, color="red")
axes[1].set_xlabel("Средний остаток ± std, баллы")
axes[1].set_title("Остатки в разрезе групп учеников")
save("20_error_analysis")

# Эксперименты по повышению точности
X_int = X.copy()
X_int["male_x_reading"] = X_int["gender_male"] * X_int["reading"]
X_int["male_x_writing"] = X_int["gender_male"] * X_int["writing"]
X_int["noprep_x_writing"] = X_int["prep_none"] * X_int["writing"]

experiments = [
    ("Linear без масштабирования", LinearRegression(), X),
    ("Linear + StandardScaler", Pipeline([("s", StandardScaler()), ("m", LinearRegression())]), X),
    ("Linear + MinMaxScaler", Pipeline([("s", MinMaxScaler()), ("m", LinearRegression())]), X),
    ("Ridge + MinMaxScaler", GridSearchCV(Pipeline([("s", MinMaxScaler()), ("m", Ridge())]),
                                          {"m__alpha": ALPHAS_RIDGE}, cv=cv,
                                          scoring="neg_root_mean_squared_error"), X),
    ("Lasso + MinMaxScaler", GridSearchCV(Pipeline([("s", MinMaxScaler()), ("m", Lasso(max_iter=100_000))]),
                                          {"m__alpha": ALPHAS_LASSO}, cv=cv,
                                          scoring="neg_root_mean_squared_error"), X),
    ("Linear + взаимодействия", Pipeline([("s", StandardScaler()), ("m", LinearRegression())]), X_int),
    ("Полином 2-й степени + Lasso",
     GridSearchCV(Pipeline([("s", StandardScaler()), ("poly", PolynomialFeatures(2, include_bias=False)),
                            ("s2", StandardScaler()), ("m", Lasso(max_iter=100_000))]),
                  {"m__alpha": np.logspace(-3, 0, 20)}, cv=cv, scoring="neg_root_mean_squared_error"), X),
    ("Random Forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                            random_state=RANDOM_STATE, n_jobs=-1), X),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=2,
                                                    random_state=RANDOM_STATE), X),
]
rows = []
for name, model, feats in experiments:
    Xtr_, Xte_ = feats.loc[X_train.index], feats.loc[X_test.index]
    c = cross_validate(model, Xtr_, y_train, cv=cv,
                       scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
    model.fit(Xtr_, y_train)
    p = model.predict(Xte_)
    rows.append({"Подход": name, "CV RMSE": -c["test_rmse"].mean(), "CV RMSE std": c["test_rmse"].std(),
                 "CV R²": c["test_r2"].mean(), "Test RMSE": np.sqrt(mean_squared_error(y_test, p)),
                 "Test R²": r2_score(y_test, p),
                 "Test MAPE, %": mean_absolute_percentage_error(y_test, p) * 100})
improve = pd.DataFrame(rows)
print("\n--- Сравнение подходов к повышению точности ---")
print(improve.round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 6))
ypos = np.arange(len(improve))
ax.barh(ypos - 0.2, improve["CV RMSE"], 0.4, xerr=improve["CV RMSE std"], label="CV RMSE ± std",
        color="#8DA0CB", error_kw={"alpha": 0.5})
ax.barh(ypos + 0.2, improve["Test RMSE"], 0.4, label="Test RMSE", color="#FC8D62")
ax.axvline(improve.loc[1, "CV RMSE"], color="black", ls="--", lw=1, label="базовая модель (CV)")
ax.set_yticks(ypos)
ax.set_yticklabels(improve["Подход"])
ax.invert_yaxis()
ax.set_xlim(4.5, 6.8)
ax.set_xlabel("RMSE, баллы")
ax.set_title("Подходы к повышению точности (чем меньше, тем лучше)")
ax.legend(loc="lower right")
save("21_improvements")
print(f"\nРАБОТА ЗАВЕРШЕНА. Графики сохранены в папке '{FIG_DIR}'.")
