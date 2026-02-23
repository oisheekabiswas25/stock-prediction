# Experiment 4: Regression Analysis for Stock Prediction

## Objective
To perform stock price prediction using Linear Regression and LSTM models.

## Detailed Procedure & Implementation Mapping

1. **Collect historical stock price data.**
   - Implemented in: `step1_download_data.py`
2. **Preprocess the data for analysis (missing data, scaling, splitting into train/test).**
   - Implemented in: `step2_preprocess.py`, `step3_scaling_split.py`
3. **Implement Linear Regression to predict future stock prices.**
   - Implemented in: `step4_linear_regression.py`
4. **Design and train an LSTM model for time-series prediction.**
   - Implemented in: `step5_lstm.py`
5. **Compare the accuracy of both models.**
   - Accuracy metrics printed in both `step4_linear_regression.py` and `step5_lstm.py` (add a summary if needed)
6. **Create a Flask backend for model predictions.**
   - Implemented in: `app.py`
7. **Build a frontend to visualize predictions using charts and graphs.**
   - Implemented in: `templates/index.html`

---

**All steps from the experiment are present in your codebase.**
- If you want a summary comparison of model accuracy, you can add a markdown or printout to aggregate results from steps 4 and 5.
- If you want a single script to run all steps in order, let me know!
