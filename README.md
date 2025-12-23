#  News Authenticity Detection

[Live Demo](https://fake-real-newz.streamlit.app/) 

[Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv)
 
---
 
## **Project Overview**

This project classifies news articles as **Fake** or **Real** using **Machine Learning** and **Deep Learning** approaches.  

The pipeline uses **TF-IDF features** for classical models and **LSTM embeddings** for deep learning to capture context and improve accuracy.  

Detecting fake news is crucial in today’s digital era to prevent misinformation from spreading.

---

## **Dataset**

- **Source:** [Fake and Real News Dataset - Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=True.csv)  
- **Process:**
  1. Two separate datasets: one for **Fake news** and one for **Real news**  
  2. Combined both into a single dataset  
  3. Cleaned the combined dataset (lowercasing, removing punctuation/special characters, stopwords, and optional lemmatization)

- **Columns:**
  | Column   | Description |
  |----------|-------------|
  | `title`  | Headline of the news article |
  | `text`   | Full content of the article |
  | `subject`| Category/subject of the news |
  | `date`   | Date of publication |
  | `label`  | 0 → Fake, 1 → Real |

---

## **Project Workflow**

### 1. Data Cleaning
- Combined Fake and Real news datasets into one  
- Lowercased all text  
- Removed punctuation, special characters, and stopwords  
- Optional lemmatization  

### 2. Exploratory Data Analysis (EDA)
- WordClouds for Fake vs Real news to visualize frequent words  
- Checked class distribution  

### 3. Feature Engineering
- Combined `title + text_clean` for richer feature representation  
- Extracted **TF-IDF features** (top 5000 unigrams and bigrams)  

### 4. Classical Machine Learning Models
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|----------|--------|----------|
| Logistic Regression | ~0.99 | 0.99 | 0.99 | 0.99 |
| Random Forest | ~0.998 | 1.0 | 1.0 | 1.0 |
 
### 5. Deep Learning: LSTM with Embeddings
- Tokenized and padded sequences  
- Model architecture: **Embedding → LSTM → Dense → Sigmoid**  
- Validation Accuracy after 4 epochs: **~0.992**  

| Epoch | Train Accuracy | Validation Accuracy |
|-------|----------------|------------------|
| 1 | 0.9086 | 0.9858 |
| 2 | 0.9861 | 0.9850 |
| 3 | 0.9886 | 0.9878 |
| 4 | 0.9900 | 0.9916 |

---



