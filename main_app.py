import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from sklearn.linear_model import LinearRegression
from streamlit import download_button
from io import StringIO
import json
import time

st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon=":mortar_board:",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stExpander {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    h1 {
        color: #2C3E50;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    h2 {
        color: #34495E;
        font-size: 1.8rem;
        margin-top: 2rem;
    }
    .stSidebar .sidebar-content {
        background-color: #2C3E50;
    }
    .icon-header {
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .icon-header img {
        width: 36px;
        height: 36px;
    }
    .icon-header h2 {
        margin: 0;
        font-size: 24px;
        line-height: 1;
    }
            
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3, 
    .main .block-container h4 {
        border-bottom: 2px solid #000000;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    /* Exclude headings in the sidebar */
    .sidebar .block-container h1, 
    .sidebar .block-container h2, 
    .sidebar .block-container h3, 
    .sidebar .block-container h4 {
        border-bottom: none;
        padding-bottom: 0;
        margin-bottom: 0;
    }
    .stButton > button {
        margin-left: 5px;
    }
    .main .block-container h2, 
    .main .block-container h3 {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

@st.cache_data
def data_cleaning():
    data = pd.read_csv("data/Student_Marks.csv")
    return data

def sidebar():
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .sidebar .sidebar-content .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    .sidebar .sidebar-content .stSlider > div > div > div > div > div {
        color: #1e3a8a;
    }
    .sidebar .sidebar-content label {
        font-family: 'Helvetica', 'Arial', sans-serif;
        font-size: 16px;
        font-weight: 600;
        color: #2C3E50;
    }
    .sidebar .sidebar-content .stSlider output {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        font-weight: bold;
        color: #34495E;
    }
    </style>
    """, unsafe_allow_html=True)

    lottie_animation = load_lottiefile("data/Animation - 1719940416157.json")

    with st.sidebar:
        st_lottie(lottie_animation, height=80, key="lottie_animation")
        
        st.markdown("### 📊 Input Parameters")
        data = data_cleaning()

        slider_labels = [
            ("Number of Courses", "number_courses"),
            ("Study Hours", "time_study")
        ]

        input_value = {}

        for label, key in slider_labels:
            st.markdown(f"<p style='margin-bottom: 0;'>{label}</p>", unsafe_allow_html=True)
            if key == "number_courses":
                input_value[key] = st.slider(
                    label,
                    min_value=int(0),
                    max_value=int(data[key].max()),
                    value=int(data[key].mean()),
                    step=1,
                    key=f"slider_{key}",
                    label_visibility="collapsed"
                )
            else:
                input_value[key] = st.slider(
                    label,
                    min_value=float(0),
                    max_value=float(data[key].max()),
                    value=float(data[key].mean()),
                    key=f"slider_{key}",
                    label_visibility="collapsed"
                )

        st.markdown("### 🎓 Student Marks Prediction")

        model = pickle.load(open("data/model.pkl", "rb"))
        input_array = np.array(list(input_value.values())).reshape(1, -1)
        prediction = model.predict(input_array)

        st.markdown(
            f"""
            <div style="
                color: #ffffff; 
                background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%);
                padding: 15px 20px; 
                border-radius: 15px; 
                text-align: center; 
                font-size: 20px; 
                font-weight: bold; 
                box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
                border: 2px solid #2C6B41;
                margin: 20px auto;
                font-family: 'Arial', sans-serif;
                transition: all 0.3s;
            ">
                Predicted Marks: <span style="
                    color: #FFD700;
                    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
                    font-size: 28px;
                ">{prediction[0]:.2f}</span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    return input_value

def styled_table(data):
    table_css = """
    <style>
    .dataframe-container {
        width: 100%;
        overflow-x: auto;
        margin: 0; 
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 0.5rem;
    }
    .dataframe {
        width: 100%;
        border-collapse: collapse;
        font-size: 16px;
        text-align: left;
        color: #333;
    }
    .dataframe th, .dataframe td {
        padding: 0.75rem;
        border-bottom: 1px solid #ddd;
    }
    .dataframe th {
        background-color: #f2f2f2;
        font-weight: bold;
        text-transform: uppercase;
    }
    .dataframe tr:hover {
        background-color: #f5f5f5;
    }
    </style>
    """

    html_table = "<div class='dataframe-container'>"
    html_table += "<table class='dataframe'>"
    html_table += "<tr><th>Number of Courses</th><th>Study Hours</th><th>Marks</th></tr>"
    for index, row in data.iterrows():
        html_table += f"<tr><td>{row['number_courses']}</td><td>{row['time_study']}</td><td>{row['Marks']}</td></tr>"
    html_table += "</table>"
    html_table += "</div>"

    st.markdown(table_css, unsafe_allow_html=True)
    st.markdown(html_table, unsafe_allow_html=True)

def main():
    input_data = sidebar()
    data = data_cleaning()

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    ">
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: 1rem;
        ">
            <img src="https://img.icons8.com/fluency/96/000000/student-center.png" alt="Student icon" style="width: 60px; height: 60px; margin-right: 15px;">
            <h1 style="
                color: white;
                font-size: 3rem;
                font-weight: bold;
                margin: 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            ">
                Student Marks Predictor
            </h1>
        </div>
        <p style="
            color: #e3f2fd;
            font-size: 1.3rem;
            font-style: italic;
            max-width: 700px;
            margin: 1rem auto;
        ">
            An interactive tool to predict student performance based on study habits
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📈 Prediction Summary")
        st.markdown("\n")
        st.info(f"Number of Courses: {input_data['number_courses']}")
        st.info(f"Study Hours: {input_data['time_study']:.2f}")
        
        model = pickle.load(open("data/model.pkl", "rb"))
        input_array = np.array(list(input_data.values())).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        
        st.success(f"Predicted Marks: {prediction:.2f}")

    with col2:
        st.header("📊 Dataset Preview")
        styled_table(data.head(5))
    
    if st.checkbox("Show Full Dataset"):
            styled_table(data)

    st.markdown("\n")

    csv = data.to_csv(index=False)
    csv_bytes = csv.encode()


    st.markdown(
    """
    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
        <img src="https://img.icons8.com/color/48/000000/download--v1.png" style="width: 30px; height: 30px; margin-right: 10px; margin-top: 5px;"/>
        <div style="display: flex; flex-direction: column;">
            <h3 style="margin: 0 0 10px 0;">Download Dataset</h3>
            <div style="margin-left: -5px;">
    """, 
    unsafe_allow_html=True
)

    st.download_button(
    label="Click to Download",
    data=csv_bytes,
    file_name="student_marks_data.csv",
    mime="text/csv",
)
    st.markdown("</div></div></div>", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown(
    """
    <div style="display: flex; align-items: flex-start; margin-bottom: 1rem;">
        <img src='https://img.icons8.com/color/48/000000/combo-chart--v1.png' alt='Chart icon' style="width: 30px; height: 30px; margin-right: 10px; margin-top: 5px;"/>
        <div style="display: flex; flex-direction: column;">
            <h2 style="margin: 0 0 10px 0;">Data Visualization</h2>
            <div style="margin-left: -5px;">
    """,
    unsafe_allow_html=True
)
    
    if st.button("Generate Visualizations"):
        with st.spinner('Creating visualizations...'):
            time.sleep(1.5)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            ax1.scatter(data['time_study'], data['Marks'], alpha=0.6, color='#1e88e5', s=50)
            ax1.set_xlabel('Study Hours', fontsize=14)
            ax1.set_ylabel('Marks', fontsize=14)
            ax1.set_title('Marks vs. Study Hours', fontsize=16)
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.tick_params(axis='both', which='major', labelsize=12)
            

            ax2.hist(data['Marks'], bins=20, color='#43a047', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Marks', fontsize=14)
            ax2.set_ylabel('Frequency', fontsize=14)
            ax2.set_title('Distribution of Marks', fontsize=16)
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.tick_params(axis='both', which='major', labelsize=12)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=data['number_courses'],
                y=data['time_study'],
                z=data['Marks'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=data['Marks'],
                    colorscale='Viridis',
                    opacity=0.8
                ),
                name='Data Points'
            ))

            X = data[['number_courses', 'time_study']]
            y = data['Marks']
            model = LinearRegression().fit(X, y)

            x_range = np.linspace(data['number_courses'].min(), data['number_courses'].max(), 20)
            y_range = np.linspace(data['time_study'].min(), data['time_study'].max(), 20)
            xx, yy = np.meshgrid(x_range, y_range)

            z = model.intercept_ + model.coef_[0] * xx + model.coef_[1] * yy

            fig.add_trace(go.Surface(
                x=xx, y=yy, z=z,
                colorscale='Greys',
                opacity=0.5,
                name='Best Fit Plane'
            ))

            fig.update_layout(
                scene=dict(
                    xaxis_title='Number of Courses',
                    yaxis_title='Study Hours',
                    zaxis_title='Marks'
                ),
                height=800,
                margin=dict(r=20, b=10, l=10, t=10),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )

            st.plotly_chart(fig, use_container_width=True)
            pass

    st.markdown("</div></div></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.header("📘 About the Model")
    st.write("""
    This prediction model uses Linear Regression to estimate student marks based on the number of courses taken and study hours invested. 
    The model achieved an r2score of 0.946, indicating a strong correlation between the input features and the predicted marks.
    
    Key metrics:
    - r2 Score: 0.946
    - Mean Absolute Error (MAE): 3.08
    
    For more information on these metrics, visit the [scikit-learn documentation](https://scikit-learn.org/stable/).
    """)

    st.markdown("---")

    st.header("📞 Contact & Feedback")
    st.write("""
    We value your input! If you have any questions, suggestions, or feedback, please don't hesitate to reach out.
    
    - 🌐 [Visit our GitHub](https://github.com/arssenic)
    - ✍️ [Leave a review](https://testimonial.to/reviews37)
    """)

    st.markdown("---")
    st.caption("App Version: 1.1.0 | Model Version: 2023.08.01")

if __name__ == "__main__":
    main()
