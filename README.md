# 🏎️ Williams Racing Performance Analysis (2015–2019)

**How can past results inform current strategy at Williams Racing?**

---

## 📌 Overview

This project provides a retrospective analysis of **Williams Racing's performance between 2015–2019**, delivering **statistically backed strategic recommendations** for race outcomes, qualifying, and driver consistency.

By applying exploratory data analysis, visualization, and hypothesis testing, the project surfaces performance gaps and opportunities, offering a **data-driven lens on how an under-resourced F1 team can optimize strategy**.

---

## 🎯 Strategic Questions & KPIs

The project was structured as a **drill-down narrative**, moving from team-level outcomes to technical performance to human execution:

1. **Grid-to-Finish Performance** → Did Williams underperform across race outcomes compared to rivals?
2. **Qualifying Sector Deficits** → Were deficits larger in technical vs. power sectors?
3. **Driver Consistency** → Were rookies less consistent than veterans during races?

---

## 🔑 Key Insights

1. Williams showed the **greatest volatility** in race position changes compared to midfield rivals.
2. **Balanced sectors**, not high-downforce ones, revealed the **largest qualifying deficits**.
3. **Rookies were less consistent** than veterans — highlighting the need for structured development programs.

---

## 🛠 Deliverables

- 📑 [Final Report (PDF)](./deliverables/williams-report.pdf)
- 🎥 [Slide Deck (PDF)](./deliverables/williams-slide-deck.pdf)
- 📊 [Interactive Tableau Dashboard](https://public.tableau.com/app/profile/frank.dong6242/viz/WilliamsRacingKPI3Add-On/Dashboard1)

---

## 📂 Repository Structure
```
├── archived/                # Old drafts and unused files
├── notebooks/               # Main Jupyter notebooks (analysis + commentary)
├── plots/                   # All generated visuals
│   ├── plots1/
│   ├── plots2/
│   ├── plots3/
│   ├── selected1/
│   ├── selected2/
│   ├── selected3/
├── processed_data/          # Feature engineered datasets
├── raw_data/                # Extracted from Kaggle
├── sql/                     # BigQuery SQL scripts
├── src/                     # Supporting Python scripts
├── deliverables/            # Final report + slide deck (PDFs)
├── requirements.txt         # Dependencies
└── README.md                # Project overview
```
---

## ⚙️ Reproducibility

**Python 3.13.0**

Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/williams-performance-analysis.git
cd williams-performance-analysis
pip install -r requirements.txt
```
Then explore the notebooks in the notebooks/ directory to reproduce the analysis.

---
## 📊 Technologies Used

Python: pandas, numpy, matplotlib, seaborn, scipy
SQL: BigQuery for data extraction and transformation
Tableau: Interactive dashboard for stakeholder presentation
Jupyter: Notebooks for reproducible analysis

## 📌 License
This project is licensed under the MIT License.

🌍 Connect

💼 [LinkedIn](https://www.linkedin.com/in/-frank-dong-/)
📊 [Interactive Tableau Dashboard](https://public.tableau.com/app/profile/frank.dong6242/viz/WilliamsRacingKPI3Add-On/Dashboard1)
