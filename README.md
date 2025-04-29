# Data Learning Starter Repo

This repository contains a **10-week, hands-on study plan** designed to build your expertise in Python, Machine Learning, Natural Language Processing (NLP), and Retrieval‑Augmented Generation (RAG). You'll practice real‑world workflows using **Google Colab** and **GitHub**, creating portfolio‑worthy projects that mirror your job environment.

## Repository Structure

```
data-learning/
│
├── README.md               <- This file: overview & roadmap
├── requirements.txt        <- Python libraries used across notebooks
├── notebooks/              <- Week-by-week Jupyter notebooks
│   ├── week01_setup.ipynb
│   ├── week02_regression_models.ipynb
│   └── ...
├── data/                   <- Sample datasets (CSV, JSON)
│   └── sample_dataset.csv
├── output/                 <- Charts, model outputs, artifacts
│   └── summary_chart.png
└── projects/               <- Mini-projects and final RAG prototype
    └── rag_doc_qa/
        ├── app.py
        ├── docs/
        │   └── example_doc.pdf
        └── config.json
```

## 10-Week Study Plan

### Week 1 – Python & Pandas Refresher
- **Objectives:** Practice data wrangling in Colab (`.groupby()`, `.apply()`, `merge()`).
- **Tasks:** Build an ETL pipeline (read → clean → analyze).
- **Datasets:** Kaggle’s Netflix or Spotify datasets.
- **Deliverable:** Cleaned dataset + visual summary + README insights.

### Week 2 – Intro to Scikit‑learn & Regression
- **Objectives:** Learn train/test split and model evaluation.
- **Tasks:** Build linear regression and decision tree models.
- **Metrics:** MAE, RMSE, R².
- **Project Idea:** Predict salaries, prices, or ratings from tabular features.

### Week 3 – Classification Models & Evaluation
- **Objectives:** Explore logistic regression, RandomForest, Gradient Boosting.
- **Concepts:** Accuracy, Precision, Recall, ROC AUC, confusion matrix.
- **Project Idea:** Classify customer churn, loan default, or survival.

### Week 4 – Feature Engineering & Pipelines
- **Objectives:** Implement feature scaling, encoding, imputing, and ML pipelines.
- **Tasks:** Use Scikit‑learn’s `Pipeline`, cross‑validation, and basic hyperparameter tuning.
- **Mini‑Project:** Enhance a prior model through engineered features.

### Week 5 – Intro to Natural Language Processing (NLP)
- **Objectives:** Master tokenization, TF‑IDF, and word embeddings.
- **Tools:** `nltk`, `scikit‑learn`, or `spaCy` for text preprocessing.
- **Deliverable:** NLP notebook with feature pipeline and evaluation chart.

### Week 6 – Hugging Face & Transformers
- **Objectives:** Install and explore the `transformers` library.
- **Tasks:** Load `distilbert‑base‑uncased`, run sentiment analysis, summarization, or zero‑shot classification.
- **Mini‑Project:** Build a text classifier or summarizer using a pre‑trained model.

### Week 7 – Embeddings & FAISS Vector Search
- **Objectives:** Understand embeddings and vector search.
- **Tasks:** Generate embeddings (OpenAI or `sentence‑transformers`), index with FAISS.
- **Mini‑Project:** Create a vector database from a set of support documents.

### Week 8 – RAG Part 1: Retrieval
- **Objectives:** Use LangChain or LlamaIndex to load, chunk, and embed documents.
- **Tasks:** Retrieve relevant context from text.
- **Deliverable:** A query interface that returns top context chunks.

### Week 9 – RAG Part 2: Generation
- **Objectives:** Integrate a generative model (OpenAI or Hugging Face).
- **Tasks:** Combine user prompt + retrieved chunks to generate coherent answers.
- **Project Idea:** Q&A system over your company’s docs or a custom PDF base.

### Week 10 – Finalization & GitHub Portfolio
- **Objectives:** Polish code, notebooks, and documentation.
- **Tasks:** Push final versions; write clear `README.md` for each project.
- **Deliverable:** A public portfolio with project summaries and links to dashboards/notebooks.
