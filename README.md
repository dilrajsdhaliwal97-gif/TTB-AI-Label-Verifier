# TTB AI Label Verifier

A computer-vision powered prototype designed to assist the U.S. Department of the Treasury (TTB) in verifying alcohol beverage labels for regulatory compliance.

This prototype performs:
- OCR extraction (EasyOCR + Tesseract)
- Brand detection
- TTB class/type verification
- ABV extraction and validation
- Net contents (mL / FL OZ) extraction and validation
- Government warning detection (robust fuzzy matching)
- Multi-image batch processing
- Noise‑resistant cider label handling

---

## 🚀 Features

- **Dual OCR Engine**: EasyOCR + Tesseract fusion  
- **Robust Preprocessing**: CLAHE, sharpening, rotation scanning  
- **TTB Classifier**: Maps market terms → TTB canonical classes  
- **ABV & Volume Extraction**: Regex + fuzzy correction  
- **Gov Warning Detection**: Header, body similarity, and fallback phrase logic  
- **Streamlit UI**: Simple, fast, and non-technical  
- **Cider‑proof Volume Logic**: Auto-corrects noisy OCR (e.g., 35 → 355 mL)

---

## 🛠️ Tech Stack

- Python 3.10+
- Streamlit
- EasyOCR
- Tesseract OCR
- OpenCV
- NumPy / Pandas

---

## 📦 Installation

```bash
git clone https://github.com/dilrajsdhaliwal97-gif/TTB-AI-Label-Verifier.git
cd TTB-AI-Label-Verifier

pip install -r requirements.txt
