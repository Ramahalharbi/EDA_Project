import streamlit as st
import pandas as pd
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# 1. Setup & Configuration
st.set_page_config(page_title="Hajj Crowd Pro", layout="wide")

# Add Logo to Sidebar
try:
    logo = Image.open('images.png') 
    st.sidebar.image(logo, use_container_width=True)
except:
    st.sidebar.warning("Logo file not found.")

st.sidebar.title("Dashboard Controls")

# Sidebar: Dataset Description
with st.sidebar.expander("ℹ️ Dataset Description"):
    st.write("This analytics dashboard provides real-time crowd monitoring, health impact analysis, and predictive safety modeling for Hajj & Umrah operations.")

# Filters
st.sidebar.subheader("Global Filters")
selected_density = st.sidebar.multiselect("Filter by Crowd Density", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
time_range = st.sidebar.slider("Filter by Hour", 0, 23, (0, 23))

# 2. Data Loading & Filtering
@st.cache_data
def load_data():
    df = pd.read_csv('hajj_umrah_crowd_management_dataset.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['Hour'] = df['Timestamp'].dt.hour
    return df

raw_df = load_data()

# Apply Filters
df = raw_df[(raw_df['Crowd_Density'].isin(selected_density)) & 
            (raw_df['Hour'] >= time_range[0]) & 
            (raw_df['Hour'] <= time_range[1])]

# 3. Model Preparation
le = LabelEncoder()
y_encoded = le.fit_transform(raw_df['Crowd_Density'])
features = ['Location_Lat', 'Location_Long', 'Movement_Speed', 'Temperature', 
            'Sound_Level_dB', 'Hour', 'Queue_Time_minutes', 'Distance_Between_People_m']
model = xgb.XGBClassifier().fit(raw_df[features], y_encoded)

# 4. Main Dashboard Header
st.title("🕋 Hajj & Umrah Crowd Management Dashboard")
st.markdown("Real-time monitoring and predictive analytics for crowd safety.")

# Summary Metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", len(df))
c2.metric("Avg Speed", f"{df['Movement_Speed'].mean():.2f} m/s")
c3.metric("Peak Hour", f"{df.groupby('Hour').size().idxmax()}:00")
c4.metric("Avg Safety", f"{df['Perceived_Safety_Rating'].mean():.1f}/5")

# --- Added Data Preview Section ---
with st.expander("📊 Data Preview"):
    st.dataframe(df.head(10))

st.markdown("---")

tab1, tab2 = st.tabs(["📊 Analytics Overview", "🤖 AI Predictor & Advisor"])

# التبويب الأول: التحليل الإحصائي
with tab1:
    st.header("Comprehensive Statistical Analysis")
    
    # Row 1: Map and Area Plot
    col1, col2 = st.columns(2)
    with col1:
        fig_map = px.density_mapbox(df, lat='Location_Lat', lon='Location_Long', z='Queue_Time_minutes',
                                     radius=20, center=dict(lat=21.3, lon=39.9), zoom=12,
                                     mapbox_style="carto-positron", title="Geographical Heatmap: Waiting Times")
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col2:
        high_crowd = df[df['Crowd_Density'] == 'High'].groupby('Hour').size().reset_index(name='Count')
        fig_area = px.area(high_crowd, x="Hour", y="Count",
                           title="Crowd density flow during the day (peak hours)",
                           line_shape='spline', color_discrete_sequence=['#ff4d4d']) 
        fig_area.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1), plot_bgcolor='white')
        st.plotly_chart(fig_area, use_container_width=True)

    # Row 2: Polar Plot and Scatter
    col3, col4 = st.columns(2)
    with col3:
        # Added Polar Plot
        fig_polar = px.bar_polar(high_crowd, r="Count", theta="Hour",
                                 color="Count", template="plotly_dark",
                                 color_continuous_scale=px.colors.sequential.Plasma_r,
                                 title="Distribution of peak crowd hours (24-hour pattern)")
        fig_polar.update_layout(polar=dict(radialaxis=dict(visible=True, showticklabels=False)))
        st.plotly_chart(fig_polar, use_container_width=True)
        
    with col4:
        fig_scatter = px.scatter(df, x="Temperature", y="Health_Condition", 
                                 color="Fatigue_Level", size="Sound_Level_dB",
                                 title="The effect of heat on the health of pilgrims")
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Insight Section
    st.subheader("💡 Insights")
    st.info("**Insight 1: Time-Based Crowd (detect peak crowd hours)**\n\nPeak crowd levels tend to appear during specific hours of the day, indicating that pilgrimage activities follow temporal patterns. These peak periods can create congestion and require proactive crowd management strategies.")

    # Row 3: Line Plot and Violin
    col5, col6 = st.columns(2)
    with col5:
        df_summary = df.groupby('Crowd_Density')['Movement_Speed'].mean().reset_index()
        order = ['Low', 'Medium', 'High']
        df_summary['Crowd_Density'] = pd.Categorical(df_summary['Crowd_Density'], categories=order, ordered=True)
        df_summary = df_summary.sort_values('Crowd_Density')
        fig_line = px.line(df_summary, x="Crowd_Density", y="Movement_Speed", markers=True, 
                           title="Decreased traffic efficiency with increased crowd density", template="plotly_white")
        fig_line.update_traces(line=dict(width=4, color='#FF5733'))
        fig_line.update_layout(xaxis_title="density level", yaxis_title="Average speed (m/s)")
        st.plotly_chart(fig_line, use_container_width=True)

    with col6:
        categories = ['Low', 'Medium', 'High']
        colors = ['#00CC96', '#FFA15A', '#EF553B']
        fig_violin = go.Figure()
        for i, category in enumerate(categories):
            subset = df[df['Crowd_Density'] == category]['Perceived_Safety_Rating']
            fig_violin.add_trace(go.Violin(x=subset, line_color=colors[i], name=category, 
                                           side='positive', orientation='h', width=3))
        fig_violin.update_layout(title="The Evolution of the Sense of Security as Crowd Density Changes",
                                 xaxis_title="Perceived safety level", yaxis_title="Crowd density",
                                 template="plotly_white", violinmode='overlay')
        st.plotly_chart(fig_violin, use_container_width=True)

# التبويب الثاني: التنبؤ والتوصيات
with tab2:
    st.header("Smart Prediction System")
    col_a, col_b = st.columns(2)
    with col_a:
        crowd_feeling = st.select_slider("Visual Crowd Density", options=["Sparse", "Moderate", "Dense", "Extremely Dense"])
    with col_b:
        temp = st.slider("Temperature (°C)", 20, 50, 35)

    if st.button("Generate AI Prediction & Strategy"):
        mapping = {"Sparse": 2.5, "Moderate": 1.5, "Dense": 0.8, "Extremely Dense": 0.3}
        input_data = pd.DataFrame([[21.4, 39.8, 0.8, temp, 70, 14, 10, mapping[crowd_feeling]]], columns=features)
        res = le.inverse_transform(model.predict(input_data))[0]
        st.success(f"### Predicted Level: {res}")
        recommendations = {
            "High": {"color": "🔴", "msg": "Crowd is very high. Avoid this area, follow staff directions, stay hydrated, maintain social distance."},
            "Medium": {"color": "🟡", "msg": "Moderate crowd. Move slowly and maintain distance."},
            "Low": {"color": "🟢", "msg": "Low crowd. This is a good time to perform the activity."}
        }
        rec = recommendations.get(res)
        st.markdown(f"### {rec['color']} Recommendation:")
        st.info(rec['msg'])