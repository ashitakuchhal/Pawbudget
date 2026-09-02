import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

print("Loading datasets...")
pets = pd.read_csv("PetData.csv")
claims = pd.read_csv("ClaimData.csv")
pets["EnrollDate"] = pd.to_datetime(pets["EnrollDate"])
claims["ClaimDate"] = pd.to_datetime(claims["ClaimDate"])

data = claims.merge(
    pets[["PetId", "Species", "Breed", "PetAge", "Premium",
          "Deductible", "EnrollPath", "EnrollDate"]],
    on="PetId", how="inner"
)
data["DaysSinceEnrollment"] = (data["ClaimDate"] - data["EnrollDate"]).dt.days
data = data[(data["DaysSinceEnrollment"] >= 0) & (data["DaysSinceEnrollment"] < 730)].copy()

first_year = data[data["DaysSinceEnrollment"] < 365]
second_year = data[(data["DaysSinceEnrollment"] >= 365) & (data["DaysSinceEnrollment"] < 730)]

first_features = first_year.groupby("PetId").agg(
    PrevClaimCount=("AmountClaimed", "count"),
    PrevTotalClaimAmount=("AmountClaimed", "sum"),
    PrevAvgClaimAmount=("AmountClaimed", "mean"),
    PrevMaxClaimAmount=("AmountClaimed", "max")
).reset_index()

claimed_days = first_year.groupby("PetId")["ClaimDate"].nunique().reset_index(name="PrevClaimedDays")
first_features = first_features.merge(claimed_days, on="PetId", how="left")

second_target = second_year.groupby("PetId")["AmountClaimed"].sum().reset_index(name="NextTotalClaimAmount")

model_data = pets[["PetId", "Species", "Breed", "PetAge", "Premium", "Deductible", "EnrollPath"]].copy()
model_data = model_data.merge(first_features, on="PetId", how="left")
model_data = model_data.merge(second_target, on="PetId", how="left")

claim_features = ["PrevClaimCount", "PrevTotalClaimAmount", "PrevAvgClaimAmount", "PrevMaxClaimAmount", "PrevClaimedDays"]
model_data[claim_features] = model_data[claim_features].fillna(0)
model_data["NextTotalClaimAmount"] = model_data["NextTotalClaimAmount"].fillna(0)

age_map = {
    "8 weeks to 12 months old": 0.5, "1 year old": 1, "2 years old": 2, "3 years old": 3,
    "4 years old": 4, "5 years old": 5, "6 years old": 6, "7 years old": 7, "8 years old": 8,
    "9 years old": 9, "10 years old": 10, "11 years old": 11, "12 years old": 12,
    "13 years old": 13, "14 years old": 14, "15 years old": 15, "16 years old": 16
}
model_data["PetAgeYears"] = model_data["PetAge"].map(age_map).fillna(5)

features = ["Species", "Breed", "PetAgeYears", "Premium", "Deductible", "EnrollPath",
            "PrevClaimCount", "PrevTotalClaimAmount", "PrevAvgClaimAmount",
            "PrevMaxClaimAmount", "PrevClaimedDays"]

X = model_data[features]
y_claim = (model_data["NextTotalClaimAmount"] > 0).astype(int)
y_cost = model_data["NextTotalClaimAmount"]

categorical = ["Species", "Breed", "EnrollPath"]
numerical = ["PetAgeYears", "Premium", "Deductible", "PrevClaimCount",
             "PrevTotalClaimAmount", "PrevAvgClaimAmount", "PrevMaxClaimAmount", "PrevClaimedDays"]

preprocessor = ColumnTransformer(transformers=[
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numerical)
])

Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_claim, test_size=0.2, random_state=42, stratify=y_claim)
Xr_train, Xr_test, yr_train, yr_test = train_test_split(X, y_cost, test_size=0.2, random_state=42)

print("Training final claim probability model (60 trees, depth 10)...")
claim_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(n_estimators=60, max_depth=10, random_state=42, n_jobs=1, class_weight="balanced"))
])
claim_model.fit(Xc_train, yc_train)
joblib.dump(claim_model, "claims_probability_model.pkl", compress=3)
print("Saved claims_probability_model.pkl")

print("Training final medical cost model (60 trees, depth 10)...")
cost_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=60, max_depth=10, random_state=42, n_jobs=1))
])
cost_model.fit(Xr_train, yr_train)
joblib.dump(cost_model, "medical_cost_model.pkl", compress=3)
print("Saved medical_cost_model.pkl")

import os
print("claims_probability_model.pkl size:", os.path.getsize("claims_probability_model.pkl")/1024, "KB")
print("medical_cost_model.pkl size:", os.path.getsize("medical_cost_model.pkl")/1024, "KB")
print("DONE")
