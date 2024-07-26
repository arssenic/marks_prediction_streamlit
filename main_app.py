import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def data_cleaning():
    data = pd.read_csv("data/Student_Marks.csv")
    return data

def sidebar():
    lottie_animation = load_lottiefile("data/Animation - 1719940416157.json")

    with st.sidebar.container():
        st_lottie(lottie_animation, height=60, key="lottie_animation")

    with st.sidebar.container():
        st.markdown("### 📊 Observations")
        data = data_cleaning()

        slider_labels = [
            ("Number of Courses", "number_courses"),
            ("Study Hours", "time_study")
        ]

        input_value = {}

        for label, key in slider_labels:
            if key == "number_courses":
                input_value[key] = st.sidebar.slider(
                label,
                min_value=int(0),
                max_value=int(data[key].max()),
                value=int(data[key].mean()),
                step=1
            )

            else:
                input_value[key] = st.sidebar.slider(
                    label,
                    min_value=float(0),
                    max_value=float(data[key].max()),
                    value=float(data[key].mean())
                )

    with st.sidebar.container():
        st.markdown("### 🎓 Student Marks Prediction")

        model = pickle.load(open("data/model.pkl", "rb"))
        input_array = np.array(list(input_value.values())).reshape(1, -1)
        prediction = model.predict(input_array)

        st.write("The marks obtained by the student are:", prediction[0])

    return input_value

st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon=":student:",
    layout="wide",
    initial_sidebar_state="expanded"
)

def styled_table(data):
    table_css = """
    <style>
    .dataframe-container {
        width: 100%; /* Set width to 100% of the container */
        overflow-x: auto; /* Enable horizontal scrolling if content exceeds width */
        margin: 20px 0;
    }
    .dataframe {
        width: 100%; /* Set table width to 100% */
        border-collapse: collapse;
        font-size: 18px;
        text-align: left;
        color: #333;
        border-radius: 0.5rem; /* Add curved corners to the table */
        overflow: hidden; /* Ensure curved corners are visible */
    }
    .dataframe th, .dataframe td {
        padding: 1rem; /* Set padding to 1rem */
        border: 1px solid #ddd;
    }
    .dataframe th {
        background-color: #CCCCCE; /* Set background color to #CCCCCE */
        font-size: 20px;
        border-radius: 0.5rem; /* Add curved corners to the table headers */
    }
    .dataframe tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    .dataframe tr:hover {
        background-color: #f1f1f1;
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


input_data = sidebar()
data = data_cleaning()

header_container = st.container()
with header_container:
    col1, col2 = st.columns([2, 16])
    col1.image('data/predictive-chart.png', width=70)
    col2.markdown("<h1 style='text-decoration: underline;'>Student Marks Predictor</h1>", unsafe_allow_html=True)

st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>An interactive environment to easily predict the performance of a student.</div>",
        unsafe_allow_html=True)

st.markdown("\n")

main_container = st.container()
with main_container:
    with st.expander("Show Data"):
        styled_table(data)
        col1, col2 = st.columns([1, 10])
        col1.image('data/research.png', width=70)
        col2.markdown("<h1 style='text-decoration: underline;'>Additional Insights</h1>", unsafe_allow_html=True)
        if st.checkbox("Show Summary Statistics"):
            col1, col2 = st.columns([1, 1.15])
            with col1:
                st.markdown("<div style='padding: 0.5rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>Data Summary:</div>",
                    unsafe_allow_html=True)
                st.markdown("\n")
                st.write(data.describe())
            with col2:
                st.markdown("<div style='padding: 0.5rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>Prediction vs Actual:</div>",
                    unsafe_allow_html=True)
                st.markdown("\n")
                X_test = pd.read_csv('data/X_test.csv')
                y_test = pd.read_csv('data/y_test.csv')
                model = pickle.load(open("data/model.pkl", "rb"))
                y_pred = model.predict(X_test)

                fig, ax = plt.subplots()
                ax.scatter(y_test, y_pred, alpha=0.5, color='red')
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
                ax.set_xlabel('Actual Marks')
                ax.set_ylabel('Predicted Marks')
                ax.set_title('Predicted vs Actual Marks')
                ax.grid(True)
                ax.legend(['Actual Marks', 'Predicted Marks'])
                ax.set_facecolor('#f9f9f9')
                for spine in ax.spines.values():
                    spine.set_edgecolor('#f9f9f9')
                    spine.set_linewidth(1.5)
                    spine.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
                st.pyplot(fig)

    if 'show_visualization' not in st.session_state:
        st.session_state.show_visualization = False

    if st.button("📉 Data Visualization"):
        st.session_state.show_visualization = True

    if st.session_state.show_visualization:
        col1, col2, col3, col4 = st.columns([0.22,1.8,0.22,1.8])

        with col1:
            col1.image('data/scatter-graph.png', width=35)
        with col2:
            st.markdown("<div style='padding: 0.5rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>Scatter Plot:</div>",
                    unsafe_allow_html=True)
            st.markdown("\n")
            fig, ax = plt.subplots()
            ax.scatter(data['time_study'], data['Marks'], alpha=0.5, color='blue')
            ax.set_xlabel('Study Hours')
            ax.set_ylabel('Marks')
            ax.set_title('Marks vs. Study Hours')
            ax.grid(True)
            ax.set_facecolor('#f9f9f9')
            for spine in ax.spines.values():
                spine.set_edgecolor('#f9f9f9')
                spine.set_linewidth(1.5)
                spine.set_path_effects([pe.withStroke(linewidth=2, foreground='black')])
            st.pyplot(fig)

        with col3:
            col3.image('data/growth.png', width=35)
        with col4:
            st.markdown("<div style='padding: 0.5rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>Line Chart:</div>",
                    unsafe_allow_html=True)
            st.markdown("\n")
            st.line_chart(data)

        col1, col2 = st.columns([0.18, 3])
        with col1:
            st.image('data/3d-modeling.png', width=40)
        with col2:
            st.markdown(
                "<div style='padding: 0.5rem; padding-right: 2rem; background-color: #CCCCCE; border-radius: 0.5rem;'>3D Plot with Best Fit Plane:</div>",
                unsafe_allow_html=True
            )

        fig = go.Figure()

        fig.add_trace(go.Scatter3d(
            x=data['number_courses'],
            y=data['time_study'],
            z=data['Marks'],
            mode='markers',
            marker=dict(size=5, color='blue', opacity=0.8)
        ))

        model = pickle.load(open("data/model.pkl", "rb"))
        xx, yy = np.meshgrid(np.linspace(data['number_courses'].min(), data['number_courses'].max(), 10),
                             np.linspace(data['time_study'].min(), data['time_study'].max(), 10))
        zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        fig.add_trace(go.Surface(x=xx, y=yy, z=zz, colorscale='Viridis', opacity=0.5))

        fig.update_layout(
            scene=dict(
                xaxis_title='Number of Courses',
                yaxis_title='Study Hours',
                zaxis_title='Marks'
            ),
        )

        st.plotly_chart(fig)

st.markdown("---")
footer_container = st.container()
with footer_container:
    col1, col2 = st.columns([1.2, 10])

    col1.image('data/info.png', width=55)
    col2.markdown("<h1 style='text-decoration: underline; font-size: 2rem; margin: 0;'>About</h1>",
                  unsafe_allow_html=True)

st.markdown(
    "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>This app uses an ML model to predict "
    "the marks of a student by taking number of courses and study hours as input.The prediction model is built using data of students and leverages machine learning techniques."
    "</div>",
    unsafe_allow_html=True)

st.markdown("\n")
main_container=st.container()
with main_container:
    if st.checkbox("Click here to read more"):
        st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>The machine learning model was trained"
        " on the Linear Regression algorithm and gave the following results:</h1>",
        unsafe_allow_html=True)

        st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>1. The r2 score of the model was 0.946."
        " For more results on r2 score, you can stick to the scikit-learn website <a href='https://scikit-learn.org/stable/' target='_blank'>scikit-learn</a>.</h1>",
        unsafe_allow_html=True)

        st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>2. The mean absolute error(MAE) was recorded about 3.08.</h1>",
        unsafe_allow_html=True)

st.markdown("\n")
st.markdown("\n")
footer2_container = st.container()
with footer2_container:
    col1, col2 = st.columns([1.2, 10])

    col1.image('data/manual.png', width=55)
    col2.markdown("<h1 style='text-decoration: underline; font-size: 2rem; margin: 0;'>User Guide</h1>",
                  unsafe_allow_html=True)

    st.markdown("""
        **Step-by-Step Guide to Use the App:**""")
    st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>1. Navigate to the Sidebar: On the left side of the screen, you will see a sidebar containing input sliders and other options.</div>",
        unsafe_allow_html=True)
    st.markdown("\n")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.image('data/down-arrow.png', width=50)
    st.markdown("\n")
    st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>2. Input Values: Use the sliders to set the number of courses and study hours. These inputs will be used to predict the student's marks.</div>",
        unsafe_allow_html=True)
    st.markdown("\n")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.image('data/down-arrow.png', width=50)
    st.markdown("\n")
    st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>3. View Prediction: The predicted marks will be displayed below the sliders as soon as you adjust the inputs.</div>",
        unsafe_allow_html=True)
    st.markdown("\n")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.image('data/down-arrow.png', width=50)
    st.markdown("\n")
    st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>4. Show Data: If you wish to see the underlying data, click on the Show Data"
        " checkbox. You can also view summary statistics and a prediction vs. actual marks plot.</div>",
        unsafe_allow_html=True)
    st.markdown("\n")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.image('data/down-arrow.png', width=50)
    st.markdown("\n")
    st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>5. Data Visualization: Click on the 📉 Data Visualization button to generate various visualizations such as scatter plots, line charts, and 3D plots.</div>",
        unsafe_allow_html=True)
    st.markdown("\n")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.image('data/down-arrow.png', width=50)
    st.markdown("\n")
    st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>6. Additional Insights: If the data is displayed, you can view additional insights by checking the Show Summary Statistics checkbox.</div>",
        unsafe_allow_html=True)
    st.markdown("\n")
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        st.image('data/down-arrow.png', width=50)
    st.markdown("\n")
    st.markdown(
        "<div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>7. About & Contact: At the bottom of the page, you will find sections providing information about the app and contact details for any queries or feedback.</div>",
        unsafe_allow_html=True)

st.markdown("\n")
st.markdown("\n")
st.markdown("\n")
footer1_container = st.container()
with footer1_container:
    col1, col2 = st.columns([1.2, 10])

    col1.image('data/mail.png', width=55)
    col2.markdown("<h1 style='text-decoration: underline; font-size: 2rem; margin: 0;'>Contact</h1>",
                  unsafe_allow_html=True)

st.markdown(
    """
    <div style='padding: 1rem; background-color: #CCCCCE; border-radius: 0.5rem ;'>
        For more information, visit the <a href='https://github.com/arssenic' target='_blank'>GitHub account</a>.
        For sharing your reviews and opinions, please refer to this link <a href='https://testimonial.to/reviews37' target='_blank'>Reviews</a>.
    </div>
    """,
    unsafe_allow_html=True
)
