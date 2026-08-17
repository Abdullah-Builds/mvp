Weblink = aku-competition.streamlit.app
# Cardiac Risk MVP

## Files
- `model.pkl` — trained production model bundle.
- `train_model.py` — reproducible training script.
- `streamlit_app.py` — Streamlit MVP that loads `model.pkl`.
- `requirements.txt` — Python dependencies.

## Run the Streamlit app
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Retrain
```bash
python train_model.py "Karachi_Cardiac_Risk_Dataset_1500.xlsx - Patient Dataset.csv"
```

## Predictions
The MVP predicts:
1. Framingham Score (pts)
2. Est. Yrs to Cardiac Event
3. 10-Yr CVD Risk (%)
4. Risk Category

The four target columns and ID-only columns are excluded from the model's input features.
Missing training values are imputed; categorical fields are one-hot encoded; numeric
predictions use Random Forest regressors and risk category uses a Random Forest classifier.

## Important
The supplied dataset appears to contain synthetic/derived outcome fields. The model therefore
learns patterns in this dataset and should not be treated as a clinically validated risk model.
=======
# mvp
>>>>>>> d2c88b61863596d00e6f5aa636086cee5b8b090f
