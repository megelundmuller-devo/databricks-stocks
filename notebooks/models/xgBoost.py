# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# dependencies = [
#   "xgboost",
#   "hyperopt",
# ]
# ///
# DBTITLE 1,Load data
from pyspark.sql.functions import col

gold_df = spark.table("stocks.gold.stocks_w_prev_returns").na.drop()

# COMMAND ----------

# MAGIC %pip install xgboost hyperopt

# COMMAND ----------

# DBTITLE 1,Create schema
spark.sql("CREATE SCHEMA IF NOT EXISTS stocks.models")

# COMMAND ----------

# DBTITLE 1,Hyperparameter tuning
import json
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import mlflow
import mlflow.xgboost

# --- Features ---
target_col = "label"
id_cols    = ["Date", "company", "label"]
feature_cols = [c for c in gold_df.columns if c not in id_cols]
print(f"Features: {len(feature_cols)}")

# --- Load with metadata columns so they're available for results display ---
pdf = gold_df.orderBy("Date").select(["Date", "company"] + feature_cols + [target_col]).toPandas()
split = int(len(pdf) * 0.8)
X_train = pdf[feature_cols].iloc[:split]
X_test  = pdf[feature_cols].iloc[split:]
y_train = pdf[target_col].iloc[:split]
y_test  = pdf[target_col].iloc[split:]
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

# --- Hyperopt search space ---
search_space = {
    "max_depth":        hp.choice("max_depth", [3, 5, 7]),
    "learning_rate":    hp.loguniform("learning_rate", -3, -1),
    "subsample":        hp.uniform("subsample", 0.6, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    "n_estimators":     hp.choice("n_estimators", [200, 300, 400]),
}

mlflow.set_experiment("/Users/marcus.egelund-muller@devoteam.com/xgb_stock_direction")

def objective(params):
    with mlflow.start_run(nested=True):
        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            early_stopping_rounds=20,
            tree_method="hist",
            n_jobs=-1,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        mlflow.log_metrics({"roc_auc": auc})
        return {"loss": -auc, "status": STATUS_OK, "model": model}

trials = Trials()
with mlflow.start_run(run_name="hyperopt_search"):
    fmin(fn=objective, space=search_space, algo=tpe.suggest,
         max_evals=20, trials=trials)

best_idx    = trials.losses().index(min(trials.losses()))
final_model = trials.results[best_idx]["model"]
print(f"Best ROC-AUC: {-min(trials.losses()):.4f}")

# COMMAND ----------

# DBTITLE 1,Calculate metrics
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. Generate probabilities (Probabilities are needed for ROC and PR)
# predict_proba returns [prob_class_0, prob_class_1] -> we take column [1]
probs = final_model.predict_proba(X_test)[:, 1]

# 2. Calculate Metrics
roc_auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)

print(f"areaUnderROC: {roc_auc}")
print(f"areaUnderPR: {pr_auc}")

# COMMAND ----------

# DBTITLE 1,Display results
probs = final_model.predict_proba(X_test)[:, 1]
preds = final_model.predict(X_test)

# Recover Date/company from the already-loaded pdf using the same positional split
meta = pdf.iloc[split:][["Date", "company"]].reset_index(drop=True)

results = meta.copy()
results["label"]            = y_test.values
results["prediction"]       = preds
results["prob_up"]          = probs
results["model_confidence"] = (results["prob_up"] - 0.5).abs() + 0.5

results.sort_values("model_confidence", ascending=False).display()

# COMMAND ----------

# DBTITLE 1,Register model
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/marcus.egelund-muller@devoteam.com/xgb_stock_direction")
mlflow.xgboost.autolog()

with mlflow.start_run(run_name="final_model") as run:
    final_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    mlflow.log_param("feature_cols", json.dumps(feature_cols))
    mlflow.xgboost.log_model(
        xgb_model=final_model,
        artifact_path="model",
        registered_model_name="stocks.models.xgb_direction_predictor",
        input_example=X_train.head(1),
    )

print(f"Run ID : {run.info.run_id}")
print("Model registered to: stocks.models.xgb_direction_predictor")
print("Next: set the 'champion' alias in Databricks > Models")
