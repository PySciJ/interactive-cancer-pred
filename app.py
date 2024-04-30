from src.component.data_preprocessing import cleaned_data
import streamlit as st
from src.component.model_predict import get_prediction
from src.component.chart import add_sidebar_get_measurements, get_radar_chart


st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon=":female-doctor:",
    layout="wide",
    initial_sidebar_state="expanded"
  )
  
with open("assets/style.css") as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

input_data = add_sidebar_get_measurements() # Populates sidebar with slider and return scaled measurements(0, 1)

with st.container():
    st.title("Breast Cancer Predictor")
    st.write("This app leverage a logistic regression model to determine whether a breast mass is benign or malignant based on the measurements it receives from cytosis lab. Analysis can be done by updating the measurements sliders in the sidebar. ")

col1, col2 = st.columns([4,1])

with col1:
    radar_chart = get_radar_chart(input_data) # Takes in scaled data from sidebar slider and create radar chart
    st.plotly_chart(radar_chart)
    
with col2:
    prediction, model, input_array_scaled = get_prediction(input_data)
    
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")
  
    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True)
        
    
    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malicious: ", model.predict_proba(input_array_scaled)[0][1])
    
    