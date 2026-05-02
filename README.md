# Advanced Analytics Algorithm Implementations

## 📌 Project Overview
This repository contains Python implementations of four advanced algorithms across distinct data analytics domains: Image Analytics, Data Analytics (Graphs), Text Analytics, and Time Series Forecasting. 

This project emphasizes algorithms that are mathematically rigorous, capable of being manually traced step-by-step, and offer innovative alternatives to standard machine learning techniques (strictly avoiding standard classification and clustering).

Team Members: 106123055, 106123141, 106123125

---

## 🚀 Implemented Algorithms

### 1. Image Analytics: Local Binary Patterns (LBP)
* **File:** `lbp_implementation.py`
* **Description:** An elegant, highly efficient texture operator used in Computer Vision. LBP labels the pixels of an image by thresholding the neighborhood of each pixel and treating the result as a binary number.
* **Why it's here:** Provides a foundational, non-neural-network approach to feature extraction that is easily verifiable through manual matrix calculations.

### 2. Data Analytics (Graph): HITS Algorithm (Hubs and Authorities)
* **File:** `hits_algorithm.py`
* **Description:** A link-analysis algorithm that rates web pages by assigning two distinct scores to each node: an "Authority" score (value of the content) and a "Hub" score (value of its outbound links).
* **Why it's here:** A sophisticated alternative to PageRank that utilizes iterative linear algebra and matrix transposition.

### 3. Text Analytics: Okapi BM25 Ranking
* **File:** `okapi_bm25.py`
* **Description:** A state-of-the-art ranking function used by search engines to estimate the relevance of documents to a given search query, utilizing document length normalization and term frequency saturation.
* **Why it's here:** Represents a significant upgrade over standard TF-IDF, widely used in modern Information Retrieval systems like Elasticsearch.

### 4. Time Series Analysis: Croston's Method
* **File:** `croston_method.py`
* **Description:** A specialized forecasting algorithm designed for "intermittent demand" data (time series with many zero values). It applies simple exponential smoothing separately to the magnitude of non-zero demands and the time intervals between them.
* **Why it's here:** Handles sparse, sporadic data much more effectively than standard continuous models like ARIMA or Holt-Winters.

---

## ⚙️ Prerequisites and Setup

These scripts are built using standard, lightweight Python libraries. To run the code, you will need Python 3.8+ and the following packages:
```bash
pip install numpy pandas matplotlib scikit-image
