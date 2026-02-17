# 🔬 Surviva - Survival Risk Prognosis & Precision Therapy Recommendation

![Streamlit App](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?style=flat-square&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

A sophisticated web application for predicting patient survival risk and providing precision medicine recommendations using advanced statistical modeling and machine learning.

## 📋 Overview

**Surviva** is an advanced clinical decision support tool that leverages the Cox Proportional Hazards model to predict survival probabilities and risk stratification for cancer patients. The app provides personalized therapy recommendations based on patient characteristics, biomarkers, and genetic profiles.

### Key Features

- **🎯 Survival Prediction**: Cox Proportional Hazards model for accurate survival probability estimation
- **📊 Risk Stratification**: Low, Medium, High risk categorization based on predictive model
- **💊 Precision Medicine**: Biomarker-guided therapy recommendations (KRAS, MSI/MMR status)
- **📈 Interactive Visualizations**: Survival curves, risk metrics, and clinical recommendations
- **🌙 Professional UI**: Dark mode interface with modern design and intuitive controls
- **🧬 Modern Branding**: Custom logo with survival curve visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ericjohsnonps2000/Surviva---Survival-Risk-Prognosis-and-Precision-Therapy-Recommendation.git
cd Surviva
```

2. **Create a virtual environment**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Prepare data** (optional for retraining):
   - Add your survival data as `survival_df.csv` in the project root
   - Expected columns: `Sex`, `tnm.stage`, `os.event`, `os.delay (months)`, `age.norm`, `kras.mut`, `msi.h`

### Running the App

**Option 1 - Windows (quick start):**
```bash
run_app.bat
```

**Option 2 - Terminal:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📦 Dependencies

- **pandas** (≥2.1.0): Data manipulation and analysis
- **numpy** (≥1.26.0): Numerical computing
- **matplotlib** (≥3.7.0): Data visualization
- **scikit-learn** (≥1.3.0): Machine learning algorithms
- **lifelines** (≥0.27.0): Survival analysis library
- **streamlit** (≥1.28.0): Interactive web app framework

## 🎨 Features & Functionality

### 1. **Patient Profile Input**
- Sex/Gender
- TNM Cancer Stage (1-4)
- Age (years)
- KRAS mutation status
- MSI/MMR status

### 2. **Survival Prediction**
- Personalized survival curve
- 1-year and 3-year survival estimates
- Risk score calculation

### 3. **Risk Stratification**
- **Low Risk**: Estimated risk ≤ 25th percentile
- **Medium Risk**: Estimated risk between 25th-75th percentiles
- **High Risk**: Estimated risk ≥ 75th percentile

### 4. **Precision Medicine Recommendations**
- Risk-stratified therapy guidance
- Age-appropriate treatment considerations
- Biomarker-directed therapy options:
  - **KRAS G12C inhibitors** (sotorasib, adagrasib)
  - **Anti-EGFR therapy** (KRAS wild-type)
  - **Immunotherapy** (MSI-H/MMR-deficient)
- Gender-specific efficacy considerations

### 5. **Dosing Considerations**
- Stage-appropriate dosing
- Age and organ function adjustments
- Sex-specific dosing recommendations

## 🏗️ Technical Architecture

### Backend
- **Model**: Cox Proportional Hazards (lifelines)
- **Training**: 70/30 train-test split, scikit-learn preprocessing
- **Evaluation**: Concordance Index (C-index)

### Frontend
- **Framework**: Streamlit (Python)
- **UI/UX**: Custom CSS styling, dark mode interface
- **Visualization**: Matplotlib with professional styling

### Data Flow
```
Patient Input → Feature Engineering → Cox Model → 
Survival Curve → Risk Stratification → Recommendations
```

## 📊 Model Details

### Cox Proportional Hazards Model
The Cox model is a semi-parametric regression model that estimates the hazard function:

```
h(t) = h₀(t) × exp(β₁X₁ + β₂X₂ + ... + βₚXₚ)
```

Where:
- `h(t)` = hazard function at time t
- `h₀(t)` = baseline hazard
- `β` = coefficients estimated from training data
- `X` = patient features/covariates

### Model Performance
- **Training C-index**: ~0.32-0.36 (acceptable for cancer survival prediction)
- **Features**: Sex, TNM Stage, Age, KRAS mutation, MSI/MMR status

## 🎯 Use Cases

1. **Clinical Decision Support**: Assist oncologists in treatment planning
2. **Patient Counseling**: Provide personalized survival estimates
3. **Research**: Analyze survival patterns across patient cohorts
4. **Trial Enrollment**: Identify high-risk patients for specialized protocols
5. **Quality Improvement**: Monitor outcome predictions vs. actual outcomes

## 🖼️ UI/UX Improvements (v2.0)

- ✅ **Night Mode Only**: Professional dark theme for reduced eye strain
- ✅ **Modern Logo**: Custom survival curve visualization with red "Surviva" text and glow effects
- ✅ **Clean Design**: Removed background patterns for professional appearance
- ✅ **High Contrast**: Improved text readability
- ✅ **Blue Accent Colors**: Professional color scheme
- ✅ **Responsive Layout**: Optimized for desktop and tablet viewing

## 🔄 Retraining the Model

To retrain the Cox model with new data:

```bash
python train_survival_models.py
```

This will generate:
- `cox_model.pkl`: Trained Cox Proportional Hazards model
- `app_config.json`: Feature coefficients and metadata
- Updated model performance metrics

## 🔐 Important Disclaimer

**This tool is for educational and research purposes only.**

- ⚠️ **Not a substitute for clinical judgment**
- ⚠️ **Consult healthcare professionals** for medical decisions
- ⚠️ **Model accuracy varies** - use with appropriate caution
- ⚠️ **Validate predictions** with current clinical guidelines
- ⚠️ **Institutional review** recommended before clinical use

## 📈 Future Improvements

- [ ] Integration with additional ML models (Random Survival Forest, Gradient Boosting)
- [ ] Multi-cohort validation studies
- [ ] Enhanced biomarker integration
- [ ] Real-time model retraining
- [ ] Mobile app version
- [ ] API endpoint for EHR integration
- [ ] Advanced visualization dashboards
- [ ] Confidence intervals for predictions

## 📚 References

1. Cox, D.R. (1972). "Regression Models and Life-Tables". Journal of the Royal Statistical Society
2. Lifelines Documentation: https://lifelines.readthedocs.io/
3. Concordance Index: Harrell, F. E., Lee, K. L., & Mark, D. B. (1996)
4. TCGA-PANCAN Dataset: https://www.cancer.gov/tcga

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Eric Johnson**
- GitHub: [@ericjohsnonps2000](https://github.com/ericjohsnonps2000)

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact: ericjohnsonps2000@gmail.com

---

**Last Updated**: February 2025  
**Version**: 2.0 (Modernized UI, Improved Logo Design, Night Mode)

## Project structure

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app |
| `train_survival_models.py` | Train Cox & RSF, save models and config |
| `plot_survival_curves.py` | Plot KM, Cox, and RSF survival curves |
| `survival_df.csv` | Input data (you add this) |
| `requirements.txt` | Python dependencies |
| `app_config.json` | Feature list, age norm, risk percentiles (generated by training) |
| `run_app.bat` / `run_training.bat` | Launchers (Windows) |

## License

MIT (or your chosen license)
