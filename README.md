# Lung Cancer Prediction Using Medical History & CT Scan Analysis

## Project Title: Lung Cancer Prediction Using Hybrid Models

## Problem Statement
Lung cancer diagnosis is a critical medical challenge that requires a combination of clinical data analysis and imaging techniques. Despite advancements in AI-driven diagnostic tools, the lack of a **unified dataset** integrating both medical history and CT scan data limits the effectiveness of stacking models. This project aims to explore the potential of using both approaches for an **aligned diagnosis**.

## Solution Overview
This project integrates two predictive models:

- **Medical history analysis**: Assesses early-stage risk based on symptoms, lifestyle choices (e.g., smoking, alcohol consumption), and genetic predisposition.
- **CT scan analysis**: Provides a definitive diagnosis by detecting visible abnormalities in lung images using deep learning techniques.

While stacking models could theoretically improve prediction accuracy, the absence of a standardized dataset prevents direct integration. Instead, we propose leveraging medical history predictions as a supplementary **contextual validation** for radiology-based findings.

## Model Performance
- **Decision Tree** for medical history analysis: **92% accuracy**
- **CNN** for CT scan images: **85.8% accuracy**
- **SVM** for CT scan images: **91.2% accuracy**
- **Stacking CNN & SVM**: **94.6% accuracy**

## Stacking Model Approach
- Instead of directly using the predictions from CNN and SVM, we extract **class probability distributions** from both models.
- These probabilities serve as input features to a **Logistic Regression-based stacking model**, allowing it to learn patterns more effectively and generalize better across cases.
- This approach ensures a **diverse representation** of classification confidence rather than binary outputs, improving the final stacked model's performance.

**DEMO: https://drive.google.com/file/d/17H7U9qEREBp4RS--4tqOMdA97izO_V79/view?usp=sharing
**

## Setup & Installation
Follow these steps to set up and run the project:

1. Clone the repository:
   ```sh
   git clone https://github.com/VishvajeetSinghM/LungCancerClassification_Antino.git
   cd LungCancerClassification_Antino
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```sh
   python app.py
   ```

## Usage Instructions
1. Start the Flask application.
2. Fill out the **medical history form** with relevant patient details (e.g., age, smoking history, symptoms).
3. Upload a **CT scan image** of the lung for analysis.
4. View the predictions for both medical history and image-based models.
5. Compare both results for a more comprehensive and aligned diagnosis.

## Stacking Model Limitations
- Stacking models improve accuracy by combining predictions, but **due to dataset limitations**, merging CT scan-based and medical history-based models is challenging.
- **Alternative Approach**: Medical history predictions can complement radiology-based findings, enhancing diagnostic confidence.

## Conclusion
Despite dataset limitations, combining these two approaches offers a **holistic assessment** of lung cancer risk. CNN-based models analyze lung images, while medical history models provide additional context. Future research should focus on developing standardized datasets to facilitate seamless model stacking for improved diagnostic precision.




