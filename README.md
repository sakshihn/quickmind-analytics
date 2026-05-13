# ⚡ QuickMind — AI-Powered Quick Commerce Delivery Intelligence System

> **End-to-End Analytics & ML Prediction Platform** for Quick Commerce Delivery Operations  
> Modeled after Blinkit · Zepto · Swiggy Instamart · Amazon Now

---

## 🚀 Live Demo

```
streamlit run app.py
→  http://localhost:8501
   Login: admin / admin123
```

---

## 📋 Project Overview

QuickMind is a **production-grade data science project** that combines real-world analytics, business intelligence, and machine learning to solve operational challenges in the quick commerce (10-minute delivery) industry.

| Attribute    | Detail                                       |
| ------------ | -------------------------------------------- |
| Dataset Size | 9,36,453 rows × 13 columns                   |
| Companies    | Blinkit, Zepto, Swiggy Instamart, Dunzo + 4  |
| Cities       | 12 major Indian metros                       |
| ML Model     | Random Forest (R²=0.9658, MAE=0.95 min)      |
| Stack        | Python, Streamlit, Plotly, Scikit-Learn, XGB |
| Deployment   | Streamlit Cloud / Local                      |

---

## 🎯 Business Problem Statement

Quick commerce platforms promise delivery under 30 minutes — a critical differentiator. Yet **SLA breaches** are frequent due to:

- Variable delivery distances
- Partner performance inconsistency
- Demand spikes in specific cities
- Product category complexity

**Solution**: An end-to-end intelligence system that:

1. Monitors SLA compliance in real-time
2. Predicts delivery time using ML (R² = 0.9658)
3. Identifies root causes of breaches
4. Generates automated business recommendations

---

## 🏗️ Architecture

```
Raw Data (CSV)
     ↓
Data Cleaning (Jupyter/VS Code)
     ↓
Cleaned Dataset ──→ EDA (eda_analysis.py)
     ↓                      ↓
Feature Engineering    Charts & Insights
     ↓
ML Pipeline (train_model.py)
  ├── Linear Regression  (baseline)
  ├── Random Forest      ✅ BEST (R²=0.97)
  └── XGBoost            (comparison)
     ↓
Saved Models (.pkl)
     ↓
Streamlit App (app.py)
  ├── Admin Login
  ├── Executive Dashboard
  ├── Analytics & EDA
  ├── Operations Dashboard
  ├── AI Predictions
  ├── Business Insights
  ├── Data Explorer
  └── Upload & Refresh
```

---

## 📁 Project Structure

```
quickcommerce/
│
├── 📂 data/
│   └── cleaned_quick_commerce.csv       # Cleaned dataset (9.4L rows)
│
├── 📂 models/
│   ├── best_model.pkl                   # Best ML model (Random Forest)
│   ├── random_forest.pkl                # Random Forest model
│   ├── xgboost.pkl                      # XGBoost model
│   ├── linear_regression.pkl            # Linear Regression baseline
│   ├── label_encoders.pkl               # Categorical encoders
│   ├── feature_names.pkl                # Feature list
│   └── model_metrics.csv                # Comparison metrics
│
├── 📂 sql/
│   └── queries.sql                      # 30+ SQL analytics queries
│
├── 📂 notebooks/
│   └── *.png                            # EDA charts (auto-generated)
│
├── 📂 assets/                           # Static assets
│
├── app.py                               # 🌐 Main Streamlit application
├── train_model.py                       # 🤖 ML training pipeline
├── eda_analysis.py                      # 📊 EDA script (generates charts)
├── requirements.txt                     # 📦 Python dependencies
└── README.md                            # 📖 This file
```

---

## 📊 Dataset Columns

| Column                  | Type  | Description                        |
| ----------------------- | ----- | ---------------------------------- |
| Order_ID                | int   | Unique order identifier            |
| Company                 | str   | Delivery platform name             |
| City                    | str   | Delivery city                      |
| Customer_Age            | int   | Age of customer (18–59)            |
| Order_Value             | int   | Order value in INR                 |
| **Delivery_Time** ✅    | int   | Minutes taken to deliver (TARGET)  |
| Distance_km             | float | Delivery distance in km            |
| Items_Count             | int   | Number of items in order           |
| Product_Category        | str   | Category of products ordered       |
| Payment_Method          | str   | Payment mode used                  |
| Customer_Rating         | int   | Rating given by customer (1–5)     |
| Discount_Applied        | int   | Whether discount was applied (0/1) |
| Delivery_Partner_Rating | int   | Delivery partner's rating (2–5)    |

---

## 🤖 Machine Learning Results

| Model             | MAE (min)  | RMSE (min) | R² Score   |
| ----------------- | ---------- | ---------- | ---------- |
| Linear Regression | 4.0765     | 5.3361     | 0.2162     |
| Random Forest ✅  | **0.9548** | **1.1144** | **0.9658** |
| XGBoost           | 0.9554     | 1.1016     | 0.9666     |

**Winner: Random Forest** selected as best model by MAE.

### Feature Importance (Top 5)

1. Distance_km — strongest predictor
2. Delivery_Partner_Rating — operator quality signal
3. Items_Count — order complexity
4. Order_Value — basket size correlation
5. Company — platform-specific logistics

---

## 📌 KPIs & Business Metrics

| KPI                 | Value    | Business Meaning                 |
| ------------------- | -------- | -------------------------------- |
| SLA Breach Rate     | ~X%      | % of orders beyond 30-min target |
| Avg Delivery Time   | ~X min   | Core performance indicator       |
| Avg Customer Rating | ~3.5 / 5 | Customer satisfaction health     |
| Revenue per Order   | ~₹X      | Monetization efficiency          |
| Partner Rating      | ~3.8 / 5 | Delivery fleet quality           |
| Discount Rate       | ~40%     | Promotion strategy indicator     |

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train ML models

```bash
python train_model.py
```

### 3. (Optional) Generate EDA charts

```bash
python eda_analysis.py
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Open: `http://localhost:8501`  
Login: `admin / admin123`

---

## ☁️ Deploy on Streamlit Cloud

1. Push this project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Add `data/` folder with the CSV
5. Click **Deploy** → share the URL!

> **Note**: Run `python train_model.py` locally first, then commit `models/` to GitHub before deploying so the pre-trained models are available in the cloud.

---

## 🔑 Admin Login

| Username | Password   | Role    |
| -------- | ---------- | ------- |
| admin    | admin123   | Admin   |
| analyst  | analyst123 | Analyst |
| viewer   | viewer123  | Viewer  |

---

## 💡 Key Business Insights

1. **SLA Breach Pattern**: Long-distance orders (>15 km) have 3× higher breach rates
2. **Partner Quality**: 5★ partners deliver 40% faster than 2★ partners
3. **Discount Impact**: Discounted orders average ~₹50 higher value
4. **City Variance**: Metro cities show 15-25% SLA breach vs 5-10% in smaller cities
5. **Category Speed**: Dairy & Fresh items have fastest delivery; Personal Care is slowest
6. **Payment Insight**: UPI/Wallet users order ~8% more frequently than COD users

---

## 🔮 Future Scope

- [ ] Real-time data streaming via Kafka/Pub-Sub
- [ ] Time-series forecasting (Prophet/LSTM) for demand prediction
- [ ] Route optimization API integration (Google Maps/Mapbox)
- [ ] Deep learning model (Tabular DNN) for higher accuracy
- [ ] Multi-language dashboard support
- [ ] Mobile-responsive PWA version
- [ ] Slack/email SLA breach alerting
- [ ] A/B test dashboard for discount strategy

---

## 📄 Resume Description

> **AI-Powered Quick Commerce Delivery Intelligence System** _(Final Year Project)_  
> Built an end-to-end analytics and prediction platform for quick commerce delivery operations using Python, Streamlit, and machine learning. Analyzed 9.4 lakh orders across 8 platforms and 12 Indian cities. Developed a Random Forest model achieving **R²=0.9658** for delivery time prediction. Created an interactive 7-page Streamlit dashboard with admin authentication, real-time KPIs, EDA charts (Plotly), SLA breach monitoring, and AI-powered predictions. Generated 30+ SQL queries for business intelligence. Delivered auto-generated business insights with actionable recommendations.  
> **Tech Stack**: Python, Pandas, NumPy, Scikit-Learn, XGBoost, Streamlit, Plotly, Matplotlib, Seaborn, SQL

---

## 🛠️ Tech Stack

```
Frontend:    Streamlit + Plotly + Custom CSS (Google Fonts)
Backend:     Python 3.10+
ML:          Scikit-Learn, XGBoost, Joblib
Data:        Pandas, NumPy
Viz:         Plotly Express, Plotly Graph Objects, Matplotlib, Seaborn
Database:    PostgreSQL (SQL queries provided)
Deployment:  Streamlit Cloud / Docker
```

---

_Built for educational and placement purposes as a final-year Data Science capstone project._
