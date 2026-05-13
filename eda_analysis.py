"""
AI-Powered Quick Commerce Delivery Intelligence System
Exploratory Data Analysis Script
Run: python eda_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import os, warnings

warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
COLORS = ["#4F46E5","#F59E0B","#10B981","#EF4444","#8B5CF6","#F97316","#06B6D4","#EC4899"]

BASE    = os.path.dirname(__file__)
DATA    = os.path.join(BASE, "data", "cleaned_quick_commerce.csv")
OUT_DIR = os.path.join(BASE, "notebooks")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  QUICK COMMERCE — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ── Load ──────────────────────────────────────────────────────────
df = pd.read_csv(DATA)
df["SLA_Breach"]    = (df["Delivery_Time"] > 30).astype(int)
df["Distance_Bin"]  = pd.cut(df["Distance_km"],
                               bins=[0,5,10,20,100],
                               labels=["Very Short","Short","Medium","Long"])
df["Age_Group"]     = pd.cut(df["Customer_Age"],
                               bins=[17,25,35,45,60],
                               labels=["18-25","26-35","36-45","46-60"])

print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumn Summary:")
print(df.describe(include="all").T[["count","unique","mean","std","min","max"]].to_string())

# ═══════════════════════════════════════════════════════════════════
# FIGURE 1 — Delivery Time Analysis
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Delivery Time Analysis", fontsize=16, fontweight="bold", y=1.01)

# 1a: Distribution
ax = axes[0,0]
ax.hist(df["Delivery_Time"], bins=40, color=COLORS[0], alpha=0.8, edgecolor="white")
ax.axvline(30, color=COLORS[3], linestyle="--", linewidth=2, label="SLA=30 min")
ax.axvline(df["Delivery_Time"].mean(), color=COLORS[1], linestyle="--",
           linewidth=2, label=f"Mean={df['Delivery_Time'].mean():.1f}")
ax.set_title("Delivery Time Distribution"); ax.legend()

# 1b: By company (violin)
ax = axes[0,1]
companies = sorted(df["Company"].unique())
data_by_co = [df[df["Company"]==c]["Delivery_Time"].values for c in companies]
parts = ax.violinplot(data_by_co, showmedians=True)
for pc, color in zip(parts["bodies"], COLORS): pc.set_facecolor(color); pc.set_alpha(0.7)
ax.axhline(30, color=COLORS[3], linestyle="--", linewidth=1.5)
ax.set_xticks(range(1, len(companies)+1))
ax.set_xticklabels(companies, rotation=30, ha="right")
ax.set_title("Delivery Time by Platform")

# 1c: SLA breach rate by company
ax = axes[0,2]
sla_co = df.groupby("Company")["SLA_Breach"].mean().mul(100).sort_values(ascending=False)
bars = ax.barh(sla_co.index, sla_co.values, color=COLORS[:len(sla_co)])
ax.axvline(10, color="gray", linestyle="--", linewidth=1)
for bar, val in zip(bars, sla_co.values):
    ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9)
ax.set_title("SLA Breach Rate by Platform"); ax.set_xlabel("Breach %")

# 1d: Distance vs Delivery Time
ax = axes[1,0]
sample = df.sample(min(5000, len(df)), random_state=42)
ax.scatter(sample["Distance_km"], sample["Delivery_Time"],
           alpha=0.25, color=COLORS[0], s=8)
m, b = np.polyfit(sample["Distance_km"], sample["Delivery_Time"], 1)
x_line = np.linspace(0, 30, 100)
ax.plot(x_line, m*x_line+b, color=COLORS[3], linewidth=2, label=f"r={sample[['Distance_km','Delivery_Time']].corr().iloc[0,1]:.3f}")
ax.axhline(30, color=COLORS[3], linestyle="--", linewidth=1)
ax.set_title("Distance vs Delivery Time"); ax.legend(); ax.set_xlabel("Distance (km)"); ax.set_ylabel("Delivery Time (min)")

# 1e: Partner rating vs avg delivery time
ax = axes[1,1]
part_del = df.groupby("Delivery_Partner_Rating")["Delivery_Time"].mean()
ax.bar(part_del.index, part_del.values, color=COLORS[:len(part_del)], edgecolor="white")
for i, val in zip(part_del.index, part_del.values):
    ax.text(i, val+0.2, f"{val:.1f}", ha="center", fontsize=9)
ax.set_title("Partner Rating → Avg Delivery Time"); ax.set_xlabel("Partner Rating"); ax.set_ylabel("Avg Delivery Time (min)")

# 1f: SLA by distance bin
ax = axes[1,2]
sla_dist = df.groupby("Distance_Bin")["SLA_Breach"].mean().mul(100)
ax.bar(sla_dist.index.astype(str), sla_dist.values,
       color=[COLORS[2], COLORS[1], COLORS[0], COLORS[3]], edgecolor="white")
for i, val in enumerate(sla_dist.values):
    ax.text(i, val+0.2, f"{val:.1f}%", ha="center", fontsize=9)
ax.set_title("SLA Breach by Distance Band"); ax.set_xlabel("Distance Band"); ax.set_ylabel("Breach %")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "01_delivery_analysis.png"), dpi=150, bbox_inches="tight")
print("\n[1/5] Saved: 01_delivery_analysis.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 2 — Revenue & Order Value
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Revenue & Order Value Analysis", fontsize=16, fontweight="bold")

# 2a: Order value distribution
ax = axes[0,0]
ax.hist(df["Order_Value"], bins=40, color=COLORS[1], alpha=0.85, edgecolor="white")
ax.axvline(df["Order_Value"].mean(), color=COLORS[3], linestyle="--",
           label=f"Mean=₹{df['Order_Value'].mean():.0f}")
ax.set_title("Order Value Distribution"); ax.set_xlabel("Order Value (₹)"); ax.legend()

# 2b: Revenue by category (horizontal bar)
ax = axes[0,1]
cat_rev = df.groupby("Product_Category")["Order_Value"].sum().sort_values(ascending=True) / 1e6
ax.barh(cat_rev.index, cat_rev.values, color=COLORS[:len(cat_rev)], edgecolor="white")
ax.set_title("Total Revenue by Category (₹M)"); ax.set_xlabel("Revenue (₹ Million)")

# 2c: Order value by company
ax = axes[0,2]
comp_aov = df.groupby("Company")["Order_Value"].mean().sort_values(ascending=False)
ax.bar(comp_aov.index, comp_aov.values, color=COLORS[:len(comp_aov)], edgecolor="white")
ax.set_xticklabels(comp_aov.index, rotation=30, ha="right")
for i, val in enumerate(comp_aov.values):
    ax.text(i, val+5, f"₹{val:.0f}", ha="center", fontsize=8)
ax.set_title("Avg Order Value by Platform"); ax.set_ylabel("Avg Order Value (₹)")

# 2d: Discount impact
ax = axes[1,0]
disc_aov = df.groupby("Discount_Applied")["Order_Value"].mean()
ax.bar(["No Discount","Discount Applied"], disc_aov.values,
       color=[COLORS[5], COLORS[2]], edgecolor="white", width=0.4)
for i, val in enumerate(disc_aov.values):
    ax.text(i, val+5, f"₹{val:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_title("Avg Order Value: Discount vs No Discount"); ax.set_ylabel("Avg Order Value (₹)")

# 2e: Payment method distribution
ax = axes[1,1]
pay_counts = df["Payment_Method"].value_counts()
ax.pie(pay_counts.values, labels=pay_counts.index, colors=COLORS[:len(pay_counts)],
       autopct="%1.1f%%", startangle=90, pctdistance=0.82)
ax.set_title("Payment Method Share")

# 2f: Items count vs order value
ax = axes[1,2]
item_aov = df.groupby("Items_Count")["Order_Value"].mean().reset_index()
ax.plot(item_aov["Items_Count"], item_aov["Order_Value"],
        color=COLORS[0], linewidth=2.5, marker="o", markersize=4)
ax.fill_between(item_aov["Items_Count"], item_aov["Order_Value"],
                alpha=0.15, color=COLORS[0])
ax.set_title("Items Count vs Avg Order Value"); ax.set_xlabel("Items Count"); ax.set_ylabel("Avg Order Value (₹)")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "02_revenue_analysis.png"), dpi=150, bbox_inches="tight")
print("[2/5] Saved: 02_revenue_analysis.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 3 — Customer Insights
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Customer Insights & Satisfaction Analysis", fontsize=16, fontweight="bold")

# 3a: Rating distribution
ax = axes[0,0]
rating_pal = ["#EF4444","#F97316","#EAB308","#84CC16","#10B981"]
rat_counts = df["Customer_Rating"].value_counts().sort_index()
ax.bar(rat_counts.index, rat_counts.values, color=rating_pal, edgecolor="white")
for i, (rat, cnt) in enumerate(rat_counts.items()):
    ax.text(rat, cnt+500, f"{cnt:,}", ha="center", fontsize=9)
ax.set_title("Customer Rating Distribution"); ax.set_xlabel("Rating (1-5)")

# 3b: Rating by company
ax = axes[0,1]
comp_rat = df.groupby("Company")["Customer_Rating"].mean().sort_values(ascending=False)
ax.barh(comp_rat.index, comp_rat.values, color=COLORS[:len(comp_rat)], edgecolor="white")
ax.axvline(comp_rat.mean(), color="gray", linestyle="--", linewidth=1.5, label="Overall Avg")
ax.set_title("Avg Customer Rating by Platform"); ax.set_xlabel("Avg Rating"); ax.legend()

# 3c: Delivery time vs customer rating
ax = axes[0,2]
rat_del = df.groupby("Customer_Rating")["Delivery_Time"].mean()
ax.bar(rat_del.index, rat_del.values, color=rating_pal, edgecolor="white")
ax.axhline(30, color=COLORS[3], linestyle="--", label="SLA=30")
for i, (r, v) in enumerate(rat_del.items()):
    ax.text(r, v+0.2, f"{v:.1f}", ha="center", fontsize=9)
ax.set_title("Avg Delivery Time by Customer Rating"); ax.set_xlabel("Customer Rating"); ax.legend()

# 3d: Age group orders
ax = axes[1,0]
age_orders = df["Age_Group"].value_counts().sort_index()
ax.bar(age_orders.index.astype(str), age_orders.values,
       color=COLORS[:len(age_orders)], edgecolor="white")
for i, val in enumerate(age_orders.values):
    ax.text(i, val+200, f"{val:,}", ha="center", fontsize=9)
ax.set_title("Orders by Customer Age Group"); ax.set_xlabel("Age Group")

# 3e: Rating by product category
ax = axes[1,1]
cat_rat = df.groupby("Product_Category")["Customer_Rating"].mean().sort_values(ascending=False)
ax.bar(cat_rat.index, cat_rat.values, color=COLORS[:len(cat_rat)], edgecolor="white")
ax.set_xticklabels(cat_rat.index, rotation=30, ha="right")
ax.set_ylim(0, 5.5)
for i, val in enumerate(cat_rat.values):
    ax.text(i, val+0.05, f"{val:.2f}", ha="center", fontsize=9)
ax.set_title("Avg Rating by Product Category"); ax.set_ylabel("Avg Rating")

# 3f: Correlation heatmap
ax = axes[1,2]
num_cols = ["Customer_Age","Order_Value","Delivery_Time","Distance_km",
            "Items_Count","Customer_Rating","Discount_Applied","Delivery_Partner_Rating"]
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink":0.8})
ax.set_title("Feature Correlation Matrix")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "03_customer_insights.png"), dpi=150, bbox_inches="tight")
print("[3/5] Saved: 03_customer_insights.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 4 — City & Platform Operations
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("City & Platform Operations Dashboard", fontsize=16, fontweight="bold")

city_stats = df.groupby("City").agg(
    Orders=("Order_ID","count"),
    Avg_Del=("Delivery_Time","mean"),
    SLA_Pct=("SLA_Breach",lambda x: x.mean()*100),
    Avg_Rating=("Customer_Rating","mean")
).reset_index()

# 4a: City order volume
ax = axes[0,0]
cs = city_stats.sort_values("Orders", ascending=True)
ax.barh(cs["City"], cs["Orders"], color=COLORS[:len(cs)], edgecolor="white")
for idx, val in enumerate(cs["Orders"]):
    ax.text(val+200, idx, f"{val:,}", va="center", fontsize=9)
ax.set_title("Order Volume by City"); ax.set_xlabel("Total Orders")

# 4b: City SLA breach
ax = axes[0,1]
cs2 = city_stats.sort_values("SLA_Pct", ascending=False)
colors_sla = [COLORS[3] if v>20 else COLORS[1] if v>10 else COLORS[2] for v in cs2["SLA_Pct"]]
ax.barh(cs2["City"], cs2["SLA_Pct"], color=colors_sla, edgecolor="white")
ax.axvline(10, color="gray", linestyle="--", linewidth=1.5, label="Target 10%")
for idx, val in enumerate(cs2["SLA_Pct"]):
    ax.text(val+0.1, idx, f"{val:.1f}%", va="center", fontsize=9)
ax.set_title("SLA Breach Rate by City"); ax.set_xlabel("Breach %"); ax.legend()

# 4c: Company × SLA breach heatmap
ax = axes[1,0]
pivot = df.pivot_table(values="SLA_Breach", index="City",
                       columns="Company", aggfunc="mean").mul(100).round(1)
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
            linewidths=0.5, ax=ax, cbar_kws={"label":"SLA Breach %"})
ax.set_title("SLA Breach Heatmap: City × Platform"); ax.set_xlabel(""); ax.set_ylabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

# 4d: Category × city order share
ax = axes[1,1]
cat_city = df.groupby(["City","Product_Category"]).size().unstack(fill_value=0)
cat_city_pct = cat_city.div(cat_city.sum(axis=1), axis=0) * 100
cat_city_pct.plot(kind="bar", stacked=True, ax=ax,
                  color=COLORS[:len(cat_city_pct.columns)], edgecolor="white")
ax.set_title("Product Category Mix by City")
ax.set_xlabel("City"); ax.set_ylabel("Share %")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.legend(title="Category", bbox_to_anchor=(1.05,1), fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "04_operations_dashboard.png"), dpi=150, bbox_inches="tight")
print("[4/5] Saved: 04_operations_dashboard.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════
# FIGURE 5 — Summary KPI Dashboard
# ═══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 6))
fig.patch.set_facecolor("#F1F5F9")

kpis = [
    ("Total Orders",       f"{len(df):,}",                    "#4F46E5"),
    ("Avg Delivery Time",  f"{df['Delivery_Time'].mean():.1f} min", "#10B981"),
    ("SLA Breach Rate",    f"{df['SLA_Breach'].mean()*100:.1f}%",   "#EF4444"),
    ("Avg Customer Rating",f"{df['Customer_Rating'].mean():.2f} ★", "#F59E0B"),
    ("Avg Order Value",    f"₹{df['Order_Value'].mean():.0f}",      "#8B5CF6"),
    ("Total Revenue",      f"₹{df['Order_Value'].sum()/1e6:.0f}M",  "#06B6D4"),
]

for i, (label, val, color) in enumerate(kpis):
    ax = fig.add_subplot(1, 6, i+1)
    ax.set_facecolor("white")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values(): spine.set_color(color); spine.set_linewidth(2)
    ax.text(0.5, 0.65, val, ha="center", va="center", fontsize=16,
            fontweight="bold", color=color, transform=ax.transAxes)
    ax.text(0.5, 0.25, label, ha="center", va="center", fontsize=9,
            color="#64748B", transform=ax.transAxes, fontweight="500")

plt.suptitle("Quick Commerce — KPI Summary Dashboard", fontsize=14,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "05_kpi_summary.png"), dpi=150, bbox_inches="tight")
print("[5/5] Saved: 05_kpi_summary.png")
plt.close()

# ── Print Key Insights ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  KEY BUSINESS INSIGHTS")
print("=" * 60)
print(f"\n  Total Orders      : {len(df):,}")
print(f"  Avg Delivery Time : {df['Delivery_Time'].mean():.1f} minutes")
print(f"  SLA Breach Rate   : {df['SLA_Breach'].mean()*100:.1f}%")
print(f"  Avg Cust. Rating  : {df['Customer_Rating'].mean():.2f}/5")
print(f"  Total Revenue     : ₹{df['Order_Value'].sum()/1e6:.1f}M")
print(f"\n  Top Platform (Revenue)  : {df.groupby('Company')['Order_Value'].sum().idxmax()}")
print(f"  Lowest SLA Breach City  : {df.groupby('City')['SLA_Breach'].mean().idxmin()}")
print(f"  Highest SLA Breach City : {df.groupby('City')['SLA_Breach'].mean().idxmax()}")
print(f"\n  Saved all charts to: notebooks/")
print("\n" + "=" * 60)