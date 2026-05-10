<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Amazon%20Recommendation%20System&fontSize=38&fontColor=ffffff&fontAlignY=38&desc=EDA%20%7C%20Clustering%20%7C%20Collaborative%20Filtering%20%7C%20ML%20Classification&descAlignY=58&descSize=15&animation=fadeIn" width="100%"/>

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

<br/>

> **An end-to-end machine learning project** on 7.8 million Amazon product ratings —  
> from raw data exploration to a working recommendation engine and 87% accuracy classifier.

<br/>

</div>

---

## 📌 Project Overview

This project builds a complete **product recommendation system** using real-world Amazon ratings data. Starting from 7.8 million raw ratings, the pipeline covers data cleaning, exploratory analysis, user clustering, collaborative filtering, and ML-based classification — all in a single, well-documented notebook.

| 📊 Dataset | 🧑‍🤝‍🧑 Users | 🛍️ Products | ⭐ Ratings | 🎯 Best Model Accuracy |
|:----------:|:-----------:|:------------:|:-----------:|:---------------------:|
| Amazon Product Ratings | 1,540 (filtered) | 39,939 | 117,315 (filtered) | **87.23%** |

---

## 🗂️ Project Structure

```
Amazon-Recommendation-System/
│
├── project_1.ipynb          # Main notebook — full pipeline
├── ratings_.csv             # Dataset (Amazon product ratings)
└── README.md
```

---

## 🔍 Workflow

### 1️⃣ Data Cleaning & Preprocessing
- Loaded 7.8M ratings with columns: `user_id`, `prod_id`, `rating`, `timestamp`
- Dropped `timestamp` — irrelevant to recommendation logic
- No missing values found
- **Filtered** to keep only:
  - Users with **50+ ratings** (active users)
  - Products with **5+ ratings** (well-rated items)
- Result: **117,315 rows | 1,540 users | 39,939 products**

---

### 2️⃣ Exploratory Data Analysis (EDA)

| Insight | Finding |
|:--------|:--------|
| Average rating | 4.01 / 5.0 |
| Median rating | 5.0 (heavily skewed) |
| 5-star ratings | **56.3%** of all ratings |
| 4+5 star combined | **82%** of all ratings |
| Rating std | 1.38 — response bias confirmed |

**Key visuals generated:**
- 📊 Rating distribution histogram
- 📦 Boxplot — ratings spread
- 🥧 Pie chart — rating breakdown
- 📈 Top 10 most-rated products (long tail problem identified)
- 🔥 Heatmap — Top 20 users × Top 20 products (sparsity visualized)

> **Insight:** Customers who bother rating usually liked the product. Unhappy customers rarely leave ratings — a classic *response bias* pattern.

---

### 3️⃣ User Clustering

Users were clustered based on **average rating** and **rating count** (scaled with StandardScaler).

| Algorithm | Silhouette Score | Notes |
|:----------|:----------------:|:------|
| KMeans (k=3) | 0.4433 | Elbow + silhouette used for k selection |
| **Hierarchical** | **0.5796** ✅ | Best clustering result |
| DBSCAN | N/A | Only 1 cluster found — data not density-based |

> **Winner:** Hierarchical Clustering with a silhouette score of **0.5796**

---

### 4️⃣ Recommendation System (Collaborative Filtering)

**Approach:** User-Based Collaborative Filtering using Cosine Similarity

```
User-Item Matrix  →  Cosine Similarity  →  Find Similar Users  →  Recommend Unrated Products
   (1540 × 39939)      (1540 × 1540)
```

**How it works:**
1. Built a **1,540 × 1,540** user similarity matrix
2. For a given user, find the **5 most similar users**
3. Recommend products those users loved but the target user hasn't rated yet
4. Results: Top 5 recommendations — **all rated 5.0** by similar users

> Cosine similarity ignores zeros (unrated items), making it ideal for sparse matrices.

---

### 5️⃣ ML Classification

Framed as a binary classification: **Liked (rating ≥ 4) = 1, Not Liked = 0**

**Features used:**
- `user_enc` — encoded user ID
- `prod_enc` — encoded product ID
- `user_avg` — user's average rating
- `prod_avg` — product's average rating

**Train/Test Split:** 80/20 → 93,852 train | 23,463 test

| Model | Accuracy | Recall (Liked) | Notes |
|:------|:--------:|:--------------:|:------|
| Decision Tree | 82.65% | 0.89 | Tends to overfit |
| **Random Forest** | **87.23%** ✅ | **0.95** | 100 trees, best overall |

> **Random Forest wins** — 100 trees voting together reduces overfitting and almost never misses a product the user would like.

---

## 📊 Results Summary

```
╔══════════════════════════════════════════════════════════╗
║               PROJECT RESULTS AT A GLANCE                ║
╠══════════════════════════════════════════════════════════╣
║  EDA         │ 117,315 ratings · 1,540 users · 39,939 products ║
║  Clustering  │ Hierarchical best → Silhouette: 0.5796         ║
║  Recommender │ Cosine Similarity · 1540×1540 matrix            ║
║  Classifier  │ Random Forest → 87.23% accuracy                 ║
╚══════════════════════════════════════════════════════════╝
```

---

## 🛠️ Tech Stack

| Category | Tools |
|:---------|:------|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (KMeans, Hierarchical, DBSCAN, Decision Tree, Random Forest) |
| Similarity | Cosine Similarity (sklearn.metrics.pairwise) |
| Environment | Jupyter Notebook |

---

## ▶️ How to Run

```bash
# 1. Clone the repository
git clone https://github.com/Shivas28/Amazon-Recommendation-System.git
cd Amazon-Recommendation-System

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook project_1.ipynb
```

> ⚠️ Make sure `ratings_.csv` is in the same directory as the notebook.

---

## 💡 Key Learnings

- **Response bias** is real — skewed ratings datasets need careful EDA before modeling
- **Data sparsity** is the #1 challenge in recommendation systems — cosine similarity handles it naturally
- **Hierarchical clustering** outperforms KMeans and DBSCAN for user behavior data
- **Random Forest** consistently beats single decision trees by reducing variance through ensemble voting
- Filtering to active users and well-rated products is critical — raw 7.8M rows → clean 117K rows

---

## 🤝 Connect

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Shivas%20Jajala-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/shivas-jajala-1a89332b8)
[![GitHub](https://img.shields.io/badge/GitHub-Shivas28-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Shivas28)
[![Email](https://img.shields.io/badge/Email-shivasjajala@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:shivasjajala@gmail.com)

</div>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=120&section=footer" width="100%"/>
