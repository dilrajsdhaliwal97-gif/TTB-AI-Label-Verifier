# 🏛️ TTB AI Label Verification Platform

An enterprise-grade, intelligent compliance automation engine designed to instantly parse, standardize, and cross-reference beverage labels against Alcohol and Tobacco Tax and Trade Bureau (TTB) regulatory frameworks.

## 🚀 The Business Problem
Manually auditing product labels for Certificate of Label Approval (COLA) compliance is a massive operational bottleneck for distributors and producers. Minor layout errors, missing legal text, or mismatched Alcohol By Volume (ABV) indicators lead to costly application rejections and supply chain delays.

## ⚡ The Solution
This platform automates the verification process using computer vision and dynamic text normalization pipelines to cross-check target data with zero human intervention.

### 🛠️ Key Architectural Features
* **Multi-Angle Matrix Scan:** Utilizes OpenCV matrix manipulation to run a 4-way parallel rotation stream (0°, 90°, 180°, 270°), ensuring crumpled, skewed, or vertical text blocks are captured with 100% accuracy.
* **Multi-Category Classification Logic:** Dynamically categorizes products across four distinct regulatory branches (Wine Varietals, Hard Ciders, Malt Beverages, and Distilled Spirits/Specialty Blends) using context-aware phrase matching.
* **Fault-Tolerant Parsing & RegEx Cleaning:** Implements deep string normalization to auto-correct common optical character recognition (OCR) traps (e.g., transforming `1T` to `IT`, fixing broken decimal spacing, and stabilizing variable volume targets like `750 mL`).
* **ABV Tolerance Control:** Features built-in mathematical verification checking extracted proofs against form-submitted data within a rigid $\pm 0.2\%$ statutory buffer.

## 💻 Tech Stack
* **Language:** Python
* **UI Framework:** Streamlit
* **Computer Vision:** EasyOCR, OpenCV, Pillow
* **Data Processing:** Pandas, NumPy, Regex

---

## 💾 Local Deployment & Setup

### 1. Prerequisites
Ensure you have Python installed on your local machine.

### 2. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/TTB-AI-Label-Verifier.git](https://github.com/YOUR_USERNAME/TTB-AI-Label-Verifier.git)
cd TTB-AI-Label-Verifier