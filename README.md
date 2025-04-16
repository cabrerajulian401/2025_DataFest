# 4th Place at ASA Datafest 2025

*🏅 CTRL ALT ELITE took **4th** out of 30+ teams (100+ participants) at ASA DataFest 2025.*

## 🏢 Project Overview

Team **CTRL ALT ELITE** created a data-driven tool that helps commercial real estate clients determine **when**, **where**, and **how much** office space to lease. We analyze trends from thousands of large office leases (≥10,000 sq ft) to provide actionable recommendations, driven by predictive modeling and an interactive UI.

Built during **DataFest 2025**, this project provides a 360° solution using:
- Predictive ML models
- Real-time dashboards
- Interactive 3D globe visualizations
- LLM-assisted reasoning

📁 GitHub Repo: [https://github.com/cabrerajulian401/2025_DataFest](https://github.com/cabrerajulian401/2025_DataFest)

---

## 💡 Features

### ✅ Interactive Dashboard
Built with **React + Flask**, allows clients to:
- Select industry, city, budget, and size
- View filtered metrics: rent trends, availability, crime, tax
- Get tailored office lease recommendations

### 📊 Predictive Recommendations
Machine learning models predict:
- **Optimal Office Size (sq ft)**
- **Best Market (city/region)**  
Based on client inputs and historic leasing patterns.

### 🌍 3D Lease Density Globe
Rendered with **Three.js**, this globe visualizes:
- Leasing activity across U.S. cities
- Industry-based clustering
- Data-driven spikes based on volume

### 🧠 LLM-Powered Summarization
Via **LangGraph + Ollama**, the system:
- Summarizes regional insights
- Answers natural language queries like  
  *"Why is Austin ideal for tech clients?"*

---

## 🧱 Tech Stack

| Layer       | Tools Used                                                                 |
|-------------|-----------------------------------------------------------------------------|
| **Frontend**| React, Three.js, CSS, HTML                                                 |
| **Backend** | Flask, Streamlit, Python, RESTful APIs                                     |
| **ML/Stats**| R (nnet, randomForest), pandas, scikit-learn                               |
| **Visualization** | Plotly, ggplot2, 3D Globe via Three.js                               |
| **LLM Layer**| Ollama (local LLM), LangGraph (workflow logic)                            |

---

## 🧑‍💻 Team Members

- **Julian Cabrera** – Full-stack development (Flask, React, 3D globe)
- **Salma El-Wakil** – Modeling and statistical analysis (R)
- **Yaw Boateng** – Data engineering, integration, preprocessing
- **Chris Ageh** – UI/UX, design, and visuals

---

## 🧪 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/cabrerajulian401/2025_DataFest.git
cd 2025_DataFest
```

### 2. Backend (Flask)
```bash
cd Backend
pip install -r requirements.txt
python app.py
```

### 3. Frontend (React)
```bash
cd frontend  # or correct folder
npm install
npm start  # opens http://localhost:3000
```

### 4. R Model Setup
```r
install.packages(c("dplyr", "ggplot2", "nnet", "randomForest"))
```

Ensure R is installed and R_HOME is set.

### 5. Optional: Streamlit App
```bash
cd app
streamlit run realestate_app.py
```

---

## 📁 File Structure

```
2025_DataFest/
├── Backend/
│   └── app.py
├── app/
│   └── realestate_app.py
├── eda/
│   └── lease_models.R
├── Savills_Data/
│   └── Leases.csv
├── Trimmed_RealEstateRecommender.mp4
├── 3-D_Lease_Density_map.mp4
├── CTRL ALT ELITE_Updated.pptx
├── datafest_flyer.jpeg
└── README.md
```

---

## 🎥 Demos

### 📽️ [Trimmed_RealEstateRecommender.mp4](./Trimmed_RealEstateRecommender.mp4)
- Streamlit UI walkthrough
- Filter industry + market → Receive city + size suggestions

### 🌐 [3-D_Lease_Density_map.mp4](./3-D_Lease_Density_map.mp4)
- Interactive globe demo
- Clickable lease clusters based on industry

---

## 📌 Notable Capabilities

- Smart filtering based on city, industry, and size
- 3D interactive mapping with zoom/pan
- Statistical model powered by **R**
- LLM-powered context generation (via **LangGraph + Ollama**)
- API endpoints for prediction/visual data
- Deployed dashboard powered by **Flask + React**

---

## 📄 Deliverables

| File                        | Description                                                  |
|-----------------------------|--------------------------------------------------------------|
| `README.md`                | Project documentation                                        |
| `realestate_app.py`        | Streamlit version of full application                        |
| `3-D_Lease_Density_map.mp4`| 3D visualization of lease locations                          |
| `Trimmed_RealEstateRecommender.mp4` | Recommender UI demo walkthrough                |
| `CTRL ALT ELITE_Updated.pptx`| Presentation deck from competition                          |
| `datafest_flyer.jpeg`      | Visual flyer summarizing the tool                           |

---

## 🏁 Summary

CTRL ALT ELITE offers a robust commercial lease recommendation system that fuses:
- Data science 🧠  
- Web development 🌐  
- AI-enhanced insights 🤖  

Proudly presented at **ASA DataFest 2025**, and awarded **4th place** for innovation and usability!

🔗 GitHub: [https://github.com/cabrerajulian401/2025_DataFest](https://github.com/cabrerajulian401/2025_DataFest)

