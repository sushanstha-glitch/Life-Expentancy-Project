
# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import matplotlib.pyplot as plt


# # Load the saved model and its metadata
# artifact = joblib.load("life_expectancy_model.joblib")

# model = artifact["model"]
# features = artifact["features"]
# model_name = artifact["model_name"]
# ranges = artifact["ranges"]

# # Display information for testing
# print("Model loaded successfully!")
# print("Model name:", model_name)
# print("Number of features:", len(features))
# print("Features:", features)


# # Configure the Streamlit page
# st.set_page_config(
#     page_title="Life Expectancy Predictor",
#     page_icon="🌍",
#     layout="wide"
# )

# # Display the app heading
# st.title("🌍 Life Expectancy Prediction")
# st.write(
#     "Predict a country's average life expectancy using "
#     "health, economic, and demographic indicators."
# )

# # Display the selected model
# st.caption(f"Model used: {model_name}")


# st.header("Enter Country Indicators")

# with st.form("prediction_form"):
#     col1, col2 = st.columns(2)

#     with col1:
#         year = st.number_input(
#             "Year", min_value=2000, max_value=2015, value=2015, step=1
#         )
#         status = st.selectbox(
#             "Country Status", ["Developed", "Developing"]
#         )
#         schooling = st.number_input(
#             "Schooling (years)", min_value=0.0, max_value=25.0, value=10.0
#         )
#         adult_mortality = st.number_input(
#             "Adult Mortality(per 1,000 population)", min_value=0.0, value=150.0
#         )
#         bmi = st.number_input(
#             "BMI index (kg/m²)", min_value=0.0, value=25.0
#         )
#         alcohol = st.number_input(
#             "Alcohol consumption (liters per capita)", min_value=0.0, value=5.0
#         )
#         hiv_aids = st.number_input(
#             "HIV/AIDS (deaths per 1,000 people)", min_value=0.0, value=0.1
#         )

#     with col2:
#         income_composition_of_resources = st.number_input(
#             "Income Composition of Resources (0–1)",
#             min_value=0.0, max_value=1.0, value=0.6
#         )
#         total_expenditure = st.number_input(
#             "Total Health Expenditure (as a percentage of GDP)", min_value=0.0, value=5.0
#         )
#         thinness_1_19_years = st.number_input(
#             "Thinness (ages 1–19)", min_value=0.0, value=5.0
#         )
#         thinness_5_9_years = st.number_input(
#             "Thinness (ages 5–9)", min_value=0.0, value=5.0
#         )
#         hepatitis_b = st.number_input(
#             "Hepatitis B coverage (as a percentage)", min_value=0.0, max_value=100.0, value=80.0
#         )
#         polio = st.number_input(
#             "Polio coverage (as a percentage)", min_value=0.0, max_value=100.0, value=85.0
#         )
#         diphtheria = st.number_input(
#             "Diphtheria coverage (as a percentage)", min_value=0.0, max_value=100.0, value=85.0
#         )

#     submitted = st.form_submit_button("Predict Life Expectancy")




import joblib
import numpy as np
import pandas as pd
import streamlit as st
 
bundle = joblib.load("life_expectancy_model.joblib")
model = bundle["model"]
features = bundle["features"]
ranges = bundle["ranges"]
 
 
def rng(col):
    r = ranges[col]
    return float(r["min"]), float(r["median"]), float(r["max"])
 
st.title("🌍 Life Expectancy Predictor")
st.sidebar.header("Country profile")
 
raw = {}   # everything the user chooses goes in here
raw["status"] = st.sidebar.selectbox("Development status", ["Developing", "Developed"])
 
for col, label in [("schooling", "Schooling (average years)"),
                   ("income_composition_of_resources", "Income composition of resources (0 to 1)"),
                   ("adult_mortality", "Adult mortality (deaths per 1000 adults)"),
                   ("hiv_aids", "HIV/AIDS (deaths per 1000 births)"),
                   ("bmi", "Average BMI"),
                   ("alcohol", "Alcohol (litres per person per year)"),
                   ("total_expenditure", "Health spending (% of government spending)")]:
    lo, med, hi = rng(col)
    raw[col] = st.sidebar.slider(label, lo, hi, med)
 
raw["vaccine_avg"] = st.sidebar.slider("Vaccination coverage (%)", 0.0, 100.0, 85.0)
 
lo, med, hi = rng("gdp")
raw["gdp"] = st.sidebar.number_input("GDP per capita (USD)", lo, hi, med)
 
with st.sidebar.expander("Advanced inputs (defaults are typical values)"):
    for col, label in [("infant_deaths", "Infant deaths"),
                       ("under_five_deaths", "Under-five deaths"),
                       ("measles", "Measles cases"),
                       ("percentage_expenditure", "Health expenditure (% of GDP per capita)")]:
        lo, med, hi = rng(col)
        raw[col] = st.number_input(label, lo, hi, med)

    lo, med, hi = rng("year")
    raw["year"] = st.slider("Year", int(lo), int(hi), int(med))

    for col, label in [("thinness_1_19_years", "Thinness, ages 10-19 (%)"),
                       ("thinness_5_9_years", "Thinness, ages 5-9 (%)")]:
        lo, med, hi = rng(col)
        raw[col] = st.slider(label, lo, hi, med)
 
 
LOG_COLS = ["measles", "infant_deaths", "under_five_deaths",
            "gdp", "percentage_expenditure", "hiv_aids"]
 
 
def build_row(raw):
    """Same feature engineering as the notebook."""
    row = {"year": raw["year"],
           "adult_mortality": raw["adult_mortality"],
           "alcohol": raw["alcohol"],
           "bmi": raw["bmi"],
           "total_expenditure": raw["total_expenditure"],
           "thinness_1_19_years": raw["thinness_1_19_years"],
           "thinness_5_9_years": raw["thinness_5_9_years"],
           "income_composition_of_resources": raw["income_composition_of_resources"],
           "schooling": raw["schooling"],
           "vaccine_avg": raw["vaccine_avg"]}
    row["status"] = 1 if raw["status"] == "Developed" else 0
    for c in LOG_COLS:
        row[c + "_log"] = np.log1p(raw[c])
    return pd.DataFrame([row])[features]      # same columns, same order as training
 
 
def predict(raw):
    return float(model.predict(build_row(raw))[0])
 
 
prediction = predict(raw)
st.metric("Predicted life expectancy", f"{prediction:.1f} years")
 
 
import matplotlib.pyplot as plt
 
st.subheader("What if one factor changes?")
options = {"Schooling": "schooling",
           "Income composition of resources": "income_composition_of_resources",
           "Adult mortality": "adult_mortality",
           "HIV/AIDS": "hiv_aids",
           "BMI": "bmi",
           "Alcohol": "alcohol",
           "Health spending": "total_expenditure"}
label = st.selectbox("Change this factor", list(options.keys()))
key = options[label]
 
lo, _, hi = rng(key)
xs = np.linspace(lo, hi, 30)          # 30 values from the lowest to the highest
ys = []
for v in xs:
    r = dict(raw)                      # copy of the current inputs
    r[key] = float(v)                  # change only this one
    ys.append(predict(r))
 
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(xs, ys)
ax.scatter([raw[key]], [prediction], color="red", zorder=3, label="Your current input")
ax.set_xlabel(label)
ax.set_ylabel("Predicted life expectancy (years)")
ax.legend()
st.pyplot(fig)
 
 
st.subheader("What matters most to the model")
est = model.named_steps["model"]
if hasattr(est, "feature_importances_"):
    vals, note = est.feature_importances_, "Random Forest feature importance"
elif hasattr(est, "coef_"):
    vals, note = np.abs(np.ravel(est.coef_)), "Size of Linear Regression coefficients (inputs are scaled)"
else:
    vals, note = None, None
 
if vals is not None:
    imp = pd.Series(vals, index=features).sort_values().tail(10)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    imp.plot(kind="barh", ax=ax2)
    ax2.set_title(note)
    st.pyplot(fig2)
else:
    st.info("A feature-importance chart is not available for this model type.")
 
 






