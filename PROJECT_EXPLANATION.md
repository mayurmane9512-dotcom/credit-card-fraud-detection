# Credit Card Fraud Detection System: Project Explanation

> Identify suspicious transactions using machine learning and visualize fraud patterns.
>
> **ML:** Random Forest, Logistic Regression · **Data mining & visualization:** fraud trends,
> transaction charts · **Difficulty:** Medium

You can use this document as the base for a project report, a presentation or viva
preparation. Setup and run instructions are in [README.md](README.md).

---

## Contents

1. [Abstract](#1-abstract)
2. [Problem statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Why fraud detection is hard](#4-why-fraud-detection-is-hard)
5. [System architecture](#5-system-architecture)
6. [Dataset](#6-dataset)
7. [Methodology](#7-methodology)
8. [Data mining & visualization: what the charts show](#8-data-mining--visualization-what-the-charts-show)
9. [Results](#9-results)
10. [The dashboard](#10-the-dashboard)
11. [Technology stack](#11-technology-stack)
12. [Code walkthrough](#12-code-walkthrough)
13. [Limitations](#13-limitations)
14. [Future scope](#14-future-scope)
15. [Viva questions and answers](#15-viva-questions-and-answers)
16. [Conclusion](#16-conclusion)
17. [References](#17-references)

---

## 1. Abstract

Credit card fraud costs banks, merchants and cardholders billions every year. Checking every
transaction by hand is impossible, so banks use machine learning to score each transaction
the moment it happens.

This project builds an end-to-end fraud detection system in Python:

- It **simulates 50,000 realistic card transactions** (2% fraud) over three months. The fraud
  follows four real-world fraud types.
- It **mines and visualizes the data** to find when, where and how fraud happens.
- It **trains two classifiers**, Logistic Regression and Random Forest, with techniques for
  highly imbalanced data.
- It **evaluates them with the right metrics** for rare events (precision, recall, F1,
  ROC-AUC, PR-AUC).
- It **puts everything in an interactive Streamlit dashboard** that scores single
  transactions or whole CSV files and explains its decisions.

The Random Forest gives the best balance. It catches **81% of frauds** and **91.5% of its
alerts are real fraud** (F1 = 0.86, PR-AUC = 0.92). Logistic Regression catches more fraud
(95%) but raises about 23 times as many false alarms.

---

## 2. Problem statement

Given the details of a card transaction (amount, merchant, time, location, recent card
activity, ...), **predict whether it is fraudulent**. Also **show the patterns** behind
fraud so analysts can understand and act on them.

This is a **binary classification** problem: the target `is_fraud` is 1 for fraud and 0 for
genuine.

---

## 3. Objectives

1. Build a realistic, labelled transaction dataset with known fraud patterns.
2. Perform exploratory data analysis (EDA) to find fraud trends and risk factors.
3. Engineer meaningful features such as *amount compared with the customer's usual spend*.
4. Train and compare **Logistic Regression** and **Random Forest** classifiers.
5. Handle **class imbalance**, since fraud is only 2% of the data.
6. Evaluate with metrics suitable for imbalanced data and study the
   **precision–recall trade-off**.
7. Deliver a **user-friendly dashboard** for exploring trends and checking transactions.

---

## 4. Why fraud detection is hard

| Challenge | What it means | How this project handles it |
|---|---|---|
| **Class imbalance** | Only ~2% of transactions are fraud (often under 0.2% in real life). A model that always says "genuine" is 98% accurate and useless. | `class_weight="balanced"`, stratified split, evaluation with precision/recall/PR-AUC instead of accuracy. |
| **Unequal error costs** | A missed fraud (false negative) loses money. A false alarm (false positive) blocks an honest customer and annoys them. | Threshold slider in the dashboard shows the trade-off. |
| **Fraud looks normal** | Criminals try to blend in. Each fraud type is unusual in only *some* features. | Models combine many features; Random Forest learns feature interactions. |
| **Honest unusual behaviour** | Customers travel abroad or make big one-off purchases. | The dataset includes such genuine transactions, which create realistic false alarms. |
| **Concept drift** | Fraud patterns change over time. | The data has a rising trend and a "fraud ring" spike; retraining takes one command. |

---

## 5. System architecture

```
 ┌──────────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
 │ 1. DATA GENERATION   │     │ 2. FEATURE ENGINEERING│     │ 3. EDA / VISUALIZATION │
 │ generate_data.py     │ ──► │ preprocessing.py      │ ──► │ analysis.py            │
 │ 50,000 transactions  │     │ hour, is_night,       │     │ visualize.py           │
 │ 4 fraud types        │     │ amount_to_avg_ratio...│     │ 11 charts (PNG)        │
 └──────────────────────┘     └───────────┬───────────┘     └────────────────────────┘
                                          │
                                          ▼
 ┌──────────────────────┐     ┌───────────────────────┐     ┌────────────────────────┐
 │ 6. PREDICTION        │     │ 5. EVALUATION         │     │ 4. MODEL TRAINING      │
 │ predictor.py         │ ◄── │ metrics.json          │ ◄── │ train_models.py        │
 │ predict.py (CLI)     │     │ confusion matrix, ROC,│     │ preprocessing pipeline │
 │ app.py (dashboard)   │     │ PR curve, importances │     │ + LogReg / RandomForest│
 └──────────────────────┘     └───────────────────────┘     └────────────────────────┘
```

`python main.py` runs steps 1–5. `streamlit run app.py` serves step 6 and all the
interactive charts.

Each saved model file holds the **full scikit-learn Pipeline** (preprocessing + classifier).
New raw transactions can go straight in, with no chance of preprocessing them differently
from the training data.

---

## 6. Dataset

### 6.1 Why synthetic data?

Real card transactions are confidential and can't be shared. The popular public Kaggle
dataset hides its features behind PCA (`V1 … V28`), so you can't draw meaningful charts like
"fraud by merchant category" or "fraud by hour" from it. This project generates its own
data instead. It is **realistic, explainable, and the same every run** (fixed random seed 42).

### 6.2 Columns

| Column | Type | Meaning |
|---|---|---|
| `transaction_id` | text | unique id (`TXN0000001`) |
| `timestamp` | datetime | date and time (1 Jan – 31 Mar 2026) |
| `customer_id` | text | cardholder (2,000 customers) |
| `card_type` | category | Visa, Mastercard, RuPay, Amex |
| `merchant_category` | category | grocery, restaurant, gas_station, online_shopping, entertainment, health, travel, electronics, clothing, utilities, jewelry, atm_withdrawal |
| `amount` | number | transaction amount |
| `customer_avg_amount` | number | the customer's usual (historical average) amount |
| `is_online` | 0/1 | online / card-not-present purchase |
| `is_international` | 0/1 | merchant is in another country |
| `distance_from_home_km` | number | distance from the customer's home |
| `time_since_last_txn_min` | number | minutes since the card's previous transaction |
| `txn_count_last_24h` | integer | transactions on the card in the previous 24 hours |
| `is_new_merchant` | 0/1 | first time this card is used at this merchant |
| `failed_attempts` | integer | wrong PIN / CVV attempts just before this transaction |
| `fraud_type` | category | *analysis only, never a model input*: fraud type, or `none` |
| `is_fraud` | 0/1 | **target label** |

Size: **50,000 rows**, **1,000 frauds (2.00%)**, total value 4.34 million, of which
**189,107 is fraud**.

### 6.3 How genuine transactions are generated

Each customer gets a usual spending level (log-normal around 60) and a card brand. Everyday
transactions spend about that amount, mostly during the day, close to home, at known
merchants, a few times a day. Two kinds of **unusual but honest** behaviour are added on
purpose, because real systems also see them:

- **Customers on holiday:** about 3% of transactions (plus all travel purchases) happen far
  from home, half of them abroad.
- **Big planned purchases:** about 3% of transactions are 3–4 times the usual amount.

### 6.4 How fraud is generated: four fraud types

| Fraud type | Share | What the criminal does | Signals |
|---|---|---|---|
| `stolen_card_online` | 37.6% | uses stolen card details online | online, new merchant, ~2.7× usual amount, electronics / online shopping, night |
| `cloned_card` | 25.5% | uses a skimmed / cloned card in a shop or ATM | far from home, often abroad, wrong PIN attempts, ATM / jewelry |
| `card_testing` | 23.2% | checks a stolen card with tiny charges | 1–10 amounts, minutes apart, many transactions in 24 h, new merchant |
| `account_takeover` | 13.7% | controls the account, buys big items | ~5× usual amount, failed CVV / log-in attempts, new merchant |

Every type is unusual in **only some** features and normal in the rest. Card testing, for
example, involves *small* amounts close to home, which on their own look like buying a
coffee. A model has to **combine** several weak signals to catch it. That is why a
non-linear model like Random Forest has an advantage.

On top of that, fraud **grows by ~80% over the three months** and has a **"fraud ring"
week (22–28 Feb)** with about twice the normal fraud. This gives the trend charts something
real to show.

---

## 7. Methodology

### 7.1 Feature engineering (`src/preprocessing.py → add_features`)

New features are derived from the raw columns:

| Feature | How | Why |
|---|---|---|
| `hour` | hour of `timestamp` | fraud peaks at night |
| `day_of_week`, `is_weekend` | from `timestamp` | weekly patterns |
| `is_night` | hour between 0 and 5 | a simple, strong risk flag |
| `amount_to_avg_ratio` | `amount / customer_avg_amount` | **how unusual** the amount is *for this customer*. 500 is normal for one person and very suspicious for another. |

### 7.2 Preprocessing pipeline (`build_preprocessor`)

A scikit-learn `ColumnTransformer` treats each group of columns differently:

| Columns | Steps | Reason |
|---|---|---|
| amounts, ratio, distance, minutes, counts, failed attempts | median imputation → `log1p` → standard scaling | These are heavily right-skewed (a few huge values). The log makes them more symmetric; scaling puts them on the same range, which Logistic Regression needs. |
| `hour` | median imputation → standard scaling | |
| yes/no flags | most-frequent imputation | already 0/1 |
| `merchant_category`, `card_type` | most-frequent imputation → **one-hot encoding** | Models need numbers; one-hot avoids a fake order between categories. `handle_unknown="ignore"` keeps prediction working if a new category appears. |

Imputation means uploaded CSV files with a few missing values can still be scored.

### 7.3 Train / test split

- **80% training (40,000 rows), 20% testing (10,000 rows).**
- **Stratified**, so both parts keep exactly 2% fraud (800 / 200 frauds).
- The test set is never used for training. It plays the role of *future, unseen* transactions.

### 7.4 Handling class imbalance

With 49 genuine transactions for every fraud, a model can minimise its error by ignoring
fraud completely. We use **class weights**:

- `class_weight="balanced"` (Logistic Regression) and `"balanced_subsample"` (Random Forest)
  make each fraud example count about 25 times more and each genuine one about half as much
  (weights = n_samples / (2 × class count)). Mistakes on fraud now cost the model as much
  in total as mistakes on genuine transactions.
- Alternatives (see [Future scope](#14-future-scope)): undersampling the majority class,
  oversampling with SMOTE.

### 7.5 Model 1: Logistic Regression

A linear model that estimates the probability of fraud:

```
z = b0 + b1·x1 + b2·x2 + ... + bn·xn
P(fraud) = sigmoid(z) = 1 / (1 + e^(−z))
```

- Each feature gets a weight (coefficient). A positive weight raises fraud risk, a negative
  one lowers it.
- Training finds the weights that minimise **log-loss** (cross-entropy).
- **Pros:** fast, simple, explainable (look at the coefficients), gives probabilities, a
  strong baseline.
- **Cons:** draws one straight boundary (a hyperplane), so it can't learn **interactions**
  like "a small amount is only suspicious *when* there are many rapid transactions".
- Settings: `max_iter=2000`, `class_weight="balanced"`.

### 7.6 Model 2: Random Forest

An **ensemble** of many decision trees:

1. **Bagging:** each tree trains on a random bootstrap sample of the rows.
2. **Random feature subsets:** at every split, a tree only considers a random subset of
   features (√n), so the trees differ from each other.
3. **Voting:** the predicted fraud probability is the average vote of all trees.

- A decision tree asks yes/no questions ("is `is_new_merchant` = 1?", "is
  `amount_to_avg_ratio` > 2.4?"). That naturally captures **non-linear patterns and
  interactions**.
- Averaging hundreds of trees cancels out the overfitting of single trees.
- **Pros:** high accuracy, handles mixed feature types and interactions, robust to outliers,
  gives feature importances.
- **Cons:** slower and bigger than Logistic Regression (about 3.5 MB vs 3 KB), less
  transparent than a single formula.
- Settings: `n_estimators=200` trees, `min_samples_leaf=2`,
  `class_weight="balanced_subsample"`, `n_jobs=-1` (all CPU cores), `random_state=42`.

### 7.7 Evaluation metrics

**Confusion matrix** (fraud = positive class):

|  | Predicted genuine | Predicted fraud |
|---|---|---|
| **Actually genuine** | TN: true negative ✅ | FP: false alarm ❌ |
| **Actually fraud** | FN: missed fraud ❌ | TP: caught fraud ✅ |

| Metric | Formula | Question it answers |
|---|---|---|
| Accuracy | (TP + TN) / all | How many predictions are right? *Misleading with imbalance.* |
| **Precision** | TP / (TP + FP) | Of the transactions we flagged, how many were fraud? |
| **Recall** (sensitivity) | TP / (TP + FN) | Of all frauds, how many did we catch? |
| **F1-score** | 2·P·R / (P + R) | One number balancing precision and recall. |
| ROC-AUC | area under the TPR-vs-FPR curve | How well does the model rank fraud above genuine? 0.5 = random, 1 = perfect. |
| **PR-AUC** (average precision) | area under the precision-vs-recall curve | Like ROC-AUC, but focused on the rare class. **The best single metric for imbalanced data.** Random guessing scores 0.02 here. |

The best model is chosen by **PR-AUC**.

---

## 8. Data mining & visualization: what the charts show

All charts are in `reports/figures/`. The dashboard has interactive versions with filters.

| Chart | Finding |
|---|---|
| **01 Class distribution** | 49,000 genuine vs 1,000 fraud (2%). The data is highly imbalanced. |
| **02 Daily fraud trend** | Fraud **rises steadily** over the quarter. A sharp **spike during 22–28 Feb** peaks at 30 frauds on 27 Feb (5.3% fraud rate vs ~2% normally). This is how an organised fraud-ring attack shows up. |
| **03 Fraud by hour** | Transaction volume peaks during the day, but the **fraud rate is highest between midnight and 5 am**: 15.2% at night vs 1.5% during the day. Criminals act while cardholders sleep. |
| **04 Fraud by category** | **ATM withdrawals (17.7%)** and **jewelry (17.4%)** have the highest fraud rates, then electronics (8.9%), online shopping (4.6%) and travel (4.0%). These goods are easy to resell. Grocery, health and utilities have none. |
| **05 Amount distribution** | The average fraud is **189** vs **85** for genuine. Fraud amounts are typically **2.4× the customer's usual spend** (median) vs 1.0× for genuine. Card testing adds a cluster of tiny amounts. |
| **06 Fraud by weekday** | Fraud rates are similar on every day (1.8–2.2%), so day of week is a **weak** signal. That is a useful negative result. |
| **07 Risk factors** | Fraud rate: online 4.9% vs in-store 0.9%; international 9.2% vs domestic 1.6%; **new merchant 12.6% vs known 0.4%**; night 15.2% vs day 1.5%. |
| **08 Correlation heatmap** | `failed_attempts` (0.31), `is_new_merchant` (0.30), `amount_to_avg_ratio` (0.23) and `is_night` (0.19) correlate most with `is_fraud`. No single feature correlates strongly, so you need several features together. |
| **09 Distance vs amount** | Fraud forms a cluster **far from home**: the cloned-card fraud. Honest travellers overlap with it, which explains some false alarms. |
| **10 Day × hour heatmap** | On every day of the week, fraud counts are highest **from 22:00 to 04:00**, even though far fewer people shop then. |
| **11 Fraud types** | Stolen-card-online is the most common (38%). **Account takeover** is the most expensive per case (avg 395). Card testing is frequent but tiny (avg 5). It matters because it **comes before** bigger fraud. |
| **12–15 Model charts** | Confusion matrices, ROC & PR curves, feature importance, metric comparison (see [section 9](#9-results)). |

**Card type** (dashboard, *Merchant & card* tab): Visa, Mastercard, RuPay and Amex all have
1.7–2.1% fraud. Card brand doesn't affect fraud risk.

---

## 9. Results

### 9.1 Test-set performance (10,000 unseen transactions, threshold 0.5)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Train time |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.964 | 0.354 | **0.950** | 0.516 | **0.994** | 0.887 | 0.1 s |
| **Random Forest** | **0.995** | **0.915** | 0.810 | **0.859** | 0.993 | **0.923** | 0.7 s |

Confusion matrices:

| | TN | FP (false alarms) | FN (missed fraud) | TP (caught fraud) |
|---|---|---|---|---|
| Logistic Regression | 9,453 | **347** | 10 | 190 |
| Random Forest | 9,785 | **15** | 38 | 162 |

### 9.2 Interpretation

- **Accuracy is misleading.** Both models score above 96%, but a model that never flags
  anything would also get 98%. Look at precision, recall and PR-AUC instead.
- **Logistic Regression** catches 95% of frauds, but only 35% of its alerts are real. It
  wrongly blocks **347 honest customers** for every 190 frauds it catches. The balanced
  class weights push its boundary towards "fraud", and being linear, it can't separate the
  fraud types cleanly.
- **Random Forest** raises only **15 false alarms**, and 91.5% of its alerts are real
  fraud. It still catches 81% of frauds. It has the **best F1 (0.86) and PR-AUC (0.92)**,
  so it is chosen as the best model.
- Both models have ROC-AUC ≈ 0.99. ROC-AUC looks great on imbalanced data even when
  precision is poor. **PR-AUC separates the models much better** (0.92 vs 0.89).

### 9.3 Choosing the threshold

Both models output a probability. The **decision threshold** decides when a transaction gets
flagged. Here is the trade-off for the Random Forest:

| Threshold | Precision | Recall | False alarms | Missed frauds | Fraud amount caught |
|---|---|---|---|---|---|
| 0.2 | 0.689 | 0.920 | 83 | 16 | 96.0% |
| **0.3** | **0.811** | **0.900** | **42** | **20** | **93.0%** |
| 0.5 (default) | 0.915 | 0.810 | 15 | 38 | 80.5% |
| 0.7 | 0.953 | 0.705 | 7 | 59 | 71.7% |
| 0.9 | 0.987 | 0.380 | 1 | 124 | 33.9% |

A bank that cares most about losses might choose **0.3**: 90% of frauds caught with
42 false alarms in 10,000 transactions. For comparison, Logistic Regression at threshold 0.9
gets 87.5% recall with **74** false alarms. The Random Forest wins at every operating point
that matters. Try this yourself with the slider on the dashboard's **Model Performance**
page.

### 9.4 Which features matter?

Random Forest feature importance (top 6): **`is_new_merchant` (0.26),
`amount_to_avg_ratio` (0.17)**, `amount` (0.08), `failed_attempts` (0.07), `is_online`
(0.06), `distance_from_home_km` (0.06).

The engineered feature **`amount_to_avg_ratio` is the second most important**. Comparing
each amount with the customer's own habits beats looking at the raw amount alone.

Logistic Regression's coefficients agree on direction. A high ratio, new merchant, ATM,
jewelry, night, online and international transactions raise risk; grocery, health and
utilities lower it. *(The raw `amount` gets a negative weight only because the model already
uses `amount_to_avg_ratio` and `customer_avg_amount`, which overlap with it. This is called
multicollinearity.)*

---

## 10. The dashboard

`streamlit run app.py` opens a 5-page web app at http://localhost:8501:

| Page | What you can do |
|---|---|
| 🏠 **Overview** | KPI cards (transactions, frauds, fraud rate, amount lost, best model PR-AUC), genuine-vs-fraud donut, daily trend, dataset preview and column descriptions. |
| 📈 **Fraud Trends** | Filter by **date range, merchant category and channel**. Six tabs: over time (daily count, rate, weekly loss), time of day (hour chart, day × hour heatmap, weekday), merchant & card (category rates, treemap of fraud money, card type), amounts (box plots, distance-vs-amount scatter), risk factors (flags, failed attempts, 24 h activity), fraud types (share, average amount, weekly trend, descriptions). All charts are interactive: hover, zoom, download. |
| 🤖 **Model Performance** | Metrics table and comparison chart, explanation of each metric, **threshold slider** that recalculates precision, recall, false alarms, fraud amount caught and the confusion matrix live, ROC & PR curves, feature importance / coefficients. |
| 🔍 **Check a Transaction** | Enter a transaction (or load the genuine / suspicious example). Both models show a **fraud-probability gauge**, a verdict and a risk level (Low < 30% ≤ Medium < 70% ≤ High), plus plain-English **red flags** ("Amount is 16.2× the customer's usual spend", "Made late at night (02:00)"). |
| 📂 **Batch Prediction** | Upload a CSV (or use the bundled 30-row demo file the models have never seen). Every row is scored, sorted by risk with a probability bar, summarised, and downloadable as CSV. |

---

## 11. Technology stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Data | NumPy, pandas |
| Machine learning | scikit-learn (Pipeline, ColumnTransformer, LogisticRegression, RandomForestClassifier, metrics) |
| Model storage | joblib |
| Static charts | Matplotlib, Seaborn |
| Interactive charts | Plotly |
| Web dashboard | Streamlit |
| Platform | Windows / macOS / Linux, no GPU or database needed |

---

## 12. Code walkthrough

| File | Responsibility |
|---|---|
| `src/config.py` | All paths and settings in one place: dataset size, fraud rate, seed, test size, threshold, colours. |
| `src/generate_data.py` | Simulates customers, genuine behaviour, the four fraud types and the time trend. Writes `data/transactions.csv` and an unlabelled `data/sample_transactions.csv`. |
| `src/preprocessing.py` | Feature engineering (`add_features`), the scikit-learn preprocessing pipeline, column validation and the stratified split. |
| `src/analysis.py` | Aggregations (fraud rate by any column, daily trend, weekly loss, day×hour matrix, fraud types, risk flags). Used by **both** the PNG charts and the dashboard, so the numbers always match. |
| `src/visualize.py` | The 11 EDA charts (Matplotlib/Seaborn). |
| `src/train_models.py` | Builds `Pipeline(preprocess → model)` for both classifiers, trains, evaluates, saves models, metrics, test predictions, feature importance and 4 model charts. |
| `src/predictor.py` | Loads a model, scores new transactions, assigns risk levels and generates red-flag explanations. |
| `main.py` | Command-line entry point. Runs data → EDA → training (or one step with `--step`). |
| `predict.py` | Command-line scoring of a CSV file. |
| `app.py` | The Streamlit dashboard. If no models exist yet, it offers to run the pipeline. |

**Flow of `python main.py`:**
`generate_data.save_datasets()` → `pd.read_csv` + `add_features()` →
`visualize.create_eda_charts()` → `train_models.train_and_evaluate()`.

---

## 13. Limitations

- **Synthetic data.** The patterns are realistic but simplified. Real fraud is rarer
  (often < 0.2%), noisier and changes faster. Scores on real data would be lower.
- **Behaviour features are given, not computed.** In a real bank, `time_since_last_txn_min`,
  `txn_count_last_24h` and `customer_avg_amount` would be calculated from each customer's
  transaction history in real time.
- **Random train/test split.** A time-based split (train on Jan–Feb, test on March) would
  be more realistic for detecting future fraud.
- **Fixed hyperparameters.** No hyperparameter search was run.
- **Probabilities are not calibrated.** Class weighting shifts the probabilities, so they
  rank transactions well but aren't exact frequencies.
- **Red flags are rule-based.** They help humans read a case. They aren't the model's actual
  reasoning.

---

## 14. Future scope

1. **More models:** XGBoost / LightGBM, Isolation Forest or autoencoders (unsupervised
   anomaly detection), neural networks.
2. **Imbalance techniques:** SMOTE oversampling, random undersampling, cost-sensitive
   thresholds chosen from the real cost of a missed fraud vs a false alarm.
3. **Hyperparameter tuning** with `GridSearchCV` / `RandomizedSearchCV` and stratified
   k-fold cross-validation.
4. **Explainable AI:** SHAP values to show *why the model* flagged each transaction.
5. **Real data:** add a loader for the Kaggle "Credit Card Fraud Detection" dataset
   (284,807 transactions, 0.17% fraud) to compare.
6. **Real-time scoring:** a REST API (FastAPI / Flask) and streaming (Kafka) to score
   transactions as they happen.
7. **Concept-drift monitoring:** track precision/recall over time and retrain automatically.
8. **Graph features:** link cards, devices and merchants to detect fraud rings.

---

## 15. Viva questions and answers

**Q1. Why is accuracy not a good metric for fraud detection?**
Fraud is rare (2% here). A model that predicts "genuine" for everything is 98% accurate
but catches no fraud. Precision, recall, F1 and PR-AUC focus on the fraud class.

**Q2. What is the difference between precision and recall?**
Precision: of the transactions we flagged, how many were fraud (false alarms lower it).
Recall: of all frauds, how many we caught (missed frauds lower it). Raising the threshold
increases precision and lowers recall.

**Q3. Which is worse, a false positive or a false negative?**
It depends on cost. A false negative loses the fraud amount. A false positive blocks an
honest customer, costs a support call and may lose the customer. Banks pick the threshold
by weighing these costs.

**Q4. How did you handle class imbalance?**
With class weights (`balanced` / `balanced_subsample`), so mistakes on the rare fraud class
cost more during training. I also used a stratified train/test split and imbalance-aware
metrics. SMOTE and undersampling are alternatives.

**Q5. How does Random Forest work?**
It trains many decision trees, each on a bootstrap sample of the rows and with a random
subset of features at each split, and averages their votes. This reduces the overfitting of
single trees (variance) and captures non-linear patterns.

**Q6. How does Logistic Regression produce a probability?**
It computes a weighted sum of the features, z, and passes it through the sigmoid function
1 / (1 + e^(−z)), which maps any number to a value between 0 and 1.

**Q7. Why did Random Forest beat Logistic Regression?**
Fraud types are unusual in only some features, so detecting them needs combinations of
signals. A small amount is only suspicious together with rapid repeated transactions.
Trees model these interactions. A linear model can't, so to catch most fraud it has to
flag many genuine transactions too.

**Q8. Why did you scale features and apply a log transform?**
Logistic Regression is sensitive to feature scale. Amounts and distances are very skewed.
`log1p` compresses the extreme values and `StandardScaler` puts all features on a
comparable scale. Random Forest doesn't need this, but it does no harm, and both models
share one preprocessing pipeline.

**Q9. What is one-hot encoding and why use it?**
It turns a category like `merchant_category` into separate 0/1 columns, one per category.
That way the model doesn't assume a false order (e.g. grocery < travel).

**Q10. What is `amount_to_avg_ratio` and why is it useful?**
The transaction amount divided by the customer's usual amount. It measures how unusual the
amount is for *this* customer, and it was the second most important feature.

**Q11. What is the difference between ROC-AUC and PR-AUC?**
ROC-AUC plots recall against the false positive rate. With many genuine transactions, the
false positive rate stays tiny even with hundreds of false alarms, so ROC-AUC looks
excellent. PR-AUC plots precision against recall, so false alarms show up directly. That
makes it the better metric for rare events.

**Q12. What is a stratified split?**
A split that keeps the same class proportions in the training and test sets (2% fraud in
both), so the test set has enough fraud and represents reality.

**Q13. What is overfitting and how do you avoid it here?**
Overfitting is memorising the training data instead of learning general patterns. We
evaluate on an unseen test set. Random Forest averages many randomised trees, and
`min_samples_leaf=2` stops trees from making leaves out of single examples.

**Q14. Why did you save the whole pipeline instead of only the model?**
New data then goes through *exactly* the same imputation, log transform, scaling and
encoding as the training data. This prevents training/serving mismatch bugs.

**Q15. What is the decision threshold and how would you choose it?**
The probability above which a transaction is flagged (default 0.5). You choose it from the
precision–recall trade-off and business costs. For example, 0.3 catches 90% of fraud with
42 false alarms per 10,000 transactions.

**Q16. What is concept drift?**
Fraud patterns change as criminals adapt. A model trained on old data slowly gets worse.
The fix is to monitor performance and retrain regularly. The rising trend and fraud-ring
spike in our data illustrate this.

**Q17. What do the red flags on the dashboard mean? Are they the model's reasons?**
They are simple human-readable rules (e.g. "transaction at 02:00", "16× usual spend") to
help an analyst review the case. The model's decision combines all features. For true
model explanations you would use SHAP.

**Q18. Why did you generate synthetic data instead of using the Kaggle dataset?**
The Kaggle features are anonymised (PCA V1–V28), so they can't be visualized or explained
in business terms ("fraud by merchant category"). Synthetic data with realistic, documented
patterns supports both the ML part and the data-visualization part, and it is reproducible.

---

## 16. Conclusion

This project shows a complete fraud detection workflow: realistic data, exploratory
analysis that turns raw transactions into insights (night-time, new-merchant, ATM/jewelry
and fraud-ring patterns), feature engineering, two machine-learning models trained for
imbalanced data, evaluation with the right metrics, and an interactive dashboard for
analysts.

The **Random Forest** is the better model (F1 0.86, PR-AUC 0.92, only 15 false alarms per
10,000 transactions). The analysis also shows that **how a transaction compares with the
customer's normal behaviour** is a stronger fraud signal than any single raw value.

---

## 17. References

1. Scikit-learn documentation: Logistic Regression, Random Forest, model evaluation.
   https://scikit-learn.org/stable/
2. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
3. Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for
   Unbalanced Classification*. IEEE SSCI. (Origin of the Kaggle credit card fraud dataset.)
4. Saito, T. & Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than
   the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLoS ONE.
5. Kaggle: Credit Card Fraud Detection dataset. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
6. Streamlit documentation. https://docs.streamlit.io
7. Plotly Python documentation. https://plotly.com/python/
