# Importing all required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet

# Load the trained ML model (saved as joblib)
model = joblib.load("model_pipeline.joblib")

# Streamlit Page Configurations
st.set_page_config(
    page_title="Carbon Footprint Estimator",  # Title for browser tab
    layout="wide"  # Use wide screen layout
)
# apply Inter font via Google Fonts
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">

    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 610 !important; /* bold for headings */
    }

    p, div, span, label {
        font-weight: 550 !important; /* normal for body */
    }
    </style>
    """, unsafe_allow_html=True)

# Title & Subtitle
st.title("🌍Track Your CO₂ Emissions ")
st.markdown(
    "Small changes, big impact--discover how your daily habits affect the planet. "
    "Your choices shape the planet--measure, visualize, and reduce your impact today!"
)
# Sidebar Input Section for User Lifestyle Data
st.sidebar.header("Enter Your Lifestyle Data")

# User inputs with sliders and dropdowns
electricity = st.sidebar.slider("Electricity Usage (kWh/month)", 50, 10000, 300)
car_km = st.sidebar.slider("Car Travel (km/month)", 0, 10000, 500)
flights_short = st.sidebar.slider("Short Flights (per month)", 0, 50, 0)
flights_long = st.sidebar.slider("Long Flights (per month)", 0, 50, 0)
diet = st.sidebar.selectbox("Diet Type", ["Vegan", "Vegetarian", "Non-Veg"])
shopping = st.sidebar.slider("Online Shopping Orders (per month)", 0, 500, 5)
water = st.sidebar.slider("Water Usage (liters/day)", 50, 20000, 200)

# Convert diet into numerical mapping for model
diet_map = {"Vegan": 0, "Vegetarian": 1, "Non-Veg": 2}

# Create input DataFrame to feed model
input_data = pd.DataFrame([{
    "electricity_usage": electricity,
    "car_km": car_km,
    "flights_short": flights_short,
    "flights_long": flights_long,
    "diet_type": diet_map[diet],
    "shopping_freq": shopping,
    "water_usage": water
}])

# Model Prediction
prediction = model.predict(input_data)[0]  # Monthly CO₂ prediction
yearly = prediction * 12  # Yearly CO₂ prediction

# Manual breakdown of emissions by category
# (coefficients are assumed multipliers)
breakdown = {
    "Electricity": electricity * 0.42,
    "Car": car_km * 0.20,
    "Flights (Short)": flights_short * 250,
    "Flights (Long)": flights_long * 1100,
    "Diet": [50, 100, 200][diet_map[diet]],  # Based on type
    "Shopping": shopping * 10,
    "Water": water * 0.0003 * 30,  # approx monthly
}
df_breakdown = pd.DataFrame(list(breakdown.items()), columns=["Category", "CO₂ Emissions"])

# Tabs for different analysis/visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Emissions Summary",
    "Category Breakdown",
    "Global Comparison",
    "Doomsday Clock",
    "Emissions Trends Over Time",
    "Tips & Downloads"
])

# Tab 1: Summary of Results
with tab1:
    st.subheader("Your Estimated Carbon Footprint")

    # Split into 2 columns for metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Monthly CO₂ Emissions", f"{prediction:,.0f} kg")
    with col2:
        st.metric("Yearly CO₂ Emissions", f"{yearly / 1000:,.2f} tonnes")

    # Line + scatter plot for emissions by category
    fig_line = px.scatter(
        df_breakdown,
        x="Category",
        y="CO₂ Emissions",
        color="CO₂ Emissions",
        size="CO₂ Emissions",
        color_continuous_scale="Viridis",
        title="Emissions by Category",
    )
    fig_line.update_traces(
        mode="lines+markers+text",
        text=df_breakdown["CO₂ Emissions"],
        textposition="top center"
    )
    fig_line.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        title_font=dict(size=24, color="black"),
        xaxis=dict(title="Activity Category", showgrid=False, zeroline=False),
        yaxis=dict(title="CO₂ Emissions (kg)", showgrid=True, gridcolor="lightgrey"),
        coloraxis_colorbar=dict(title="CO₂ (kg)")
    )
    st.plotly_chart(fig_line, use_container_width=True)
# Tab 2: Breakdown (Bar + Pie Chart)
with tab2:
    st.subheader("Breakdown of Your Emissions")

    # Sort values for bar chart
    df_sorted = df_breakdown.sort_values(by="CO₂ Emissions", ascending=True)

    # Horizontal bar chart
    fig_bar = px.bar(
        df_sorted,
        x="CO₂ Emissions", y="Category",
        orientation="h",
        title=".",
        text="CO₂ Emissions",
        color="Category",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_bar.update_traces(
        texttemplate='%{text:.0f} kg',
        textposition='outside',
        marker=dict(line=dict(color="black", width=0.7))  # bar border
    )
    fig_bar.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title=dict(x=0.5, font=dict(size=22, color="darkgreen")),
        xaxis=dict(title="CO₂ Emissions (kg)", gridcolor="lightgray"),
        yaxis=dict(title="", categoryorder="total ascending"),
        margin=dict(l=100, r=40, t=60, b=40),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Donut pie chart
    fig_pie = px.pie(
        df_breakdown,
        values="CO₂ Emissions", names="Category",
        title="Proportional Breakdown of Emissions",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_pie.update_traces(textinfo="percent+label", pull=[0.05] * len(df_breakdown))
    st.plotly_chart(fig_pie, use_container_width=True)

# Tab 3: Comparison with Global Average
with tab3:
    st.subheader("Relative Position: Your Emissions vs Global Avg")

    global_avg = 4700  # global average yearly emissions (kg/year)
    sustainable_target = 2000  # sustainable level (kg/year)

# Gauge Chart Indicator
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=yearly,
        delta={"reference": global_avg,
               "increasing": {"color": "red"},
               "decreasing": {"color": "green"}},
        gauge={
            "axis": {"range": [0, max(yearly, global_avg, sustainable_target) * 1.2]},
            "steps": [
                {"range": [0, sustainable_target], "color": "green"},
                {"range": [sustainable_target, global_avg], "color": "yellow"},
                {"range": [global_avg, max(yearly, global_avg) * 1.2], "color": "red"},
            ],
            "threshold": {"line": {"color": "black", "width": 4}, "value": yearly}
        },
        title={"text": "Your Yearly Emissions vs Global Average"}
    ))
    st.plotly_chart(fig_comp, use_container_width=True)

# Tab 4: Doomsday Clock
with tab4:
    st.subheader("Doomsday Clock")

#Simple gauge showing "90 seconds to midnight"
    fig_clock = go.Figure(go.Indicator(
        mode="gauge+number",
        value=90,
        title={"text": "Seconds to Midnight (2025)"},
        gauge={
            "axis": {"range": [0, 600]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 90], "color": "red"},
                {"range": [90, 300], "color": "orange"},
                {"range": [300, 600], "color": "green"},
            ],
        }
    ))
    st.plotly_chart(fig_clock, use_container_width=True)

    # Some explanation text
    st.write("""
    The **Doomsday Clock**, maintained by the *Bulletin of the Atomic Scientists*, 
    symbolizes how close humanity is to global catastrophe.  
    As of **2025**, it stands at **90 seconds to midnight**, the closest ever.  
    Reducing carbon emissions is one of the key steps to push the clock back.  
    """)

# Tab 5: Emissions Trends Over Time
with tab5:
    st.subheader("Emissions Trends Over Time")

    years = st.slider("Select Duration (Years)", 1, 20, 10)

    yearly_tonnes = yearly / 1000.0
    sustainable_target_tonnes = sustainable_target / 1000.0
    x_years = list(range(1, years + 1))

    yearly_series = [yearly_tonnes] * years
    cumulative_series = [yearly_tonnes * i for i in x_years]

    # Create line plots with area fills
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(
        go.Scatter(
            x=x_years, y=yearly_series,
            mode="lines+markers",
            name="Yearly Emissions (t/yr)",
            line=dict(color="#2E86DE", width=4),
            marker=dict(size=9, line=dict(width=1, color="white")),
            hovertemplate="Year %{x}<br>Yearly: %{y:.2f} t<extra></extra>",
        ),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=x_years, y=yearly_series,
            mode="lines",
            line=dict(color="rgba(46,134,222,0)"),
            fill="tozeroy",
            fillcolor="rgba(46,134,222,0.10)",
            name=None, showlegend=False,
            hoverinfo="skip",
        ),
        secondary_y=False,
    )
    fig_trend.add_trace(
        go.Scatter(
            x=x_years, y=cumulative_series,
            mode="lines+markers",
            name="Cumulative Emissions (t)",
            line=dict(color="#E67E22", width=4),
            marker=dict(size=8, line=dict(width=1, color="white")),
            fill="tozeroy",
            fillcolor="rgba(230,126,34,0.18)",
            hovertemplate="Year %{x}<br>Cumulative: %{y:.1f} t<extra></extra>",
        ),
        secondary_y=True,
    )

    # Add horizontal line for sustainable target
    fig_trend.add_hline(
        y=sustainable_target_tonnes,
        line_dash="dot",
        line_color="#27AE60",
        annotation_text="Sustainable Target (t/yr)",
        annotation_position="top left",
    )

    # Annotating last points
    fig_trend.add_annotation(
        x=x_years[-1], y=yearly_series[-1],
        text=f"{yearly_tonnes:.2f} t/yr",
        showarrow=True, arrowhead=2, yshift=10,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#2E86DE",
    )
    fig_trend.add_annotation(
        x=x_years[-1], y=cumulative_series[-1],
        xref="x", yref="y2",
        text=f"{cumulative_series[-1]:.1f} t total",
        showarrow=True, arrowhead=2, yshift=10,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#E67E22",
    )

    fig_trend.update_layout(
        title="Projected Carbon Emissions (No Lifestyle Change)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig_trend.update_xaxes(title_text="Year", showgrid=False)
    fig_trend.update_yaxes(
        title_text="Yearly CO₂ (tonnes/yr)",
        secondary_y=False, rangemode="tozero",
        gridcolor="rgba(0,0,0,0.08)"
    )
    fig_trend.update_yaxes(
        title_text="Cumulative CO₂ (tonnes)",
        secondary_y=True, rangemode="tozero",
        gridcolor="rgba(0,0,0,0.05)"
    )

    st.plotly_chart(fig_trend, use_container_width=True)

# Display summary text
    cumulative_total_t = cumulative_series[-1]
    st.markdown(
        f" Over **{years} years**, your lifestyle would emit **{cumulative_total_t:,.1f} tonnes of CO₂** in total. "
        f"Your yearly level is **{yearly_tonnes:.2f} t/yr**, versus the sustainable target of **{sustainable_target_tonnes:.2f} t/yr**."
    )

# Tab 6: Tips + PDF Report
with tab6:
    st.markdown(" Tips to Reduce Your Footprint")
    st.write("""
    -  Use energy-efficient appliances  
    -  Walk, cycle, or use public transport  
    -  Avoid unnecessary flights  
    -  Shift towards plant-based diets  
    -  Reuse, recycle, and reduce shopping waste  
    -  Conserve water daily  
    """)

    # Function to create simple PDF report
    def create_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph(" Your Carbon Footprint Report", styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Monthly CO<sub>2</sub> Emissions: {prediction:,.0f} kg", styles["Normal"]))
        story.append(Paragraph(f"Yearly CO<sub>2</sub> Emissions: {yearly / 1000:,.2f} tonnes", styles["Normal"]))
        story.append(Spacer(1, 12))

        # Add emissions table
        data = [["Category", " CO2 Emissions (kg)"]] + df_breakdown.values.tolist()
        table = Table(data)
        story.append(table)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()

    # Button to generate PDF
    if st.button("Generate PDF Report"):
        pdf_bytes = create_pdf()
        st.download_button(
            label="Download Report as PDF",
            data=pdf_bytes,
            file_name="carbon_footprint_report.pdf",
            mime="application/pdf",
        )




