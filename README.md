🌍 Carbon Footprint Estimator
Small changes, big impact — discover how your lifestyle choices affect the planet.

This project is an interactive web app built with Streamlit that allows users to estimate their monthly and yearly carbon footprint based on their lifestyle habits. It also provides visual breakdowns, global comparisons, doomsday clock awareness, and practical tips for reducing emissions.
The goal is not only to estimate emissions but also to raise awareness and encourage individuals to adopt sustainable practices.
Features:
Carbon Footprint Prediction using a trained ML model (scikit-learn)
Category-wise Emission Breakdown (electricity, travel, diet, shopping, water usage)
Pie Charts & Bar Charts for intuitive visualizations
Global Comparison with world averages and sustainable targets
Doomsday Clock Visualization to highlight climate urgency
missions Trends Over Time (What-if Simulation) — projects cumulative emissions for up to 20 years if lifestyle doesn’t change
PDF Report Generation — download a personalized emission report with data + charts
Tips Section for practical carbon reduction strategies
Dataset

The app is trained on a synthetic dataset of 5000 samples, generated via train_model.py.
Each record simulates a person’s lifestyle habits and their estimated CO₂ emissions.
Features included:
electricity_usage → Household electricity use (kWh/month)
car_km → Distance traveled by car (km/month)
flights_short → Short flights per month
flights_long → Long flights per month
diet_type → 0 = Vegan, 1 = Vegetarian, 2 = Non-Veg
shopping_freq → Online shopping orders per month

Tech Stack:
Frontend / UI: Streamlit
Data Handling: Pandas, NumPy
Visualization: Plotly, Matplotlib
Machine Learning: scikit-learn (Linear Regression, Ridge, Random Forest)
Model Persistence: joblib
Report Generation: ReportLab (PDF export)

Acknowledgements
Bulletin of the Atomic Scientists — Doomsday Clock reference
IPCC Reports — Global emission averages and sustainable targets
Streamlit Community — For making ML apps simple and beautiful

Your choices shape the future. Measure, visualize, and reduce your impact today!
water_usage → Liters/day
