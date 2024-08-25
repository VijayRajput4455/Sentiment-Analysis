import streamlit as st
from SentimentAnalysis import SentimentAnalysisModel
from streamlit_option_menu import option_menu

# Sidebar menu for navigation
with st.sidebar:
    selected = option_menu(
        menu_title="Analytics",
        options=["Sentiment Analysis"],
        icons=["emoji-heart-eyes"],  # Add relevant icon
        menu_icon="graph-up-arrow",  # Menu icon
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#f0f2f6"},
            "icon": {"color": "orange", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#ff5722"},
        }
    )

# Load the sentiment analysis model
vector_model_path = "vecter.pkl"
model_path = "NLP.pkl"
sentiment_model = SentimentAnalysisModel(vector_model_path, model_path)

if selected == "Sentiment Analysis":
    st.markdown("<h1 style='text-align: center; color: #ff5722;'>Sentiment Analysis</h1>", unsafe_allow_html=True)
    
    # Input text area with custom styling
    input_chat = st.text_area(
        "Enter your Text Here:",
        height=150,
        placeholder="Type Something to Analyze the Sentiment...",
        help="Input the Text you want to Analyze for Sentiment...."
    )

    # Predict button with custom style
    if st.button("Predict", key="predict_button", help="Click to predict the sentiment"):
        if input_chat:
            # Perform sentiment analysis
            sentiment = sentiment_model.predict_sentiment(input_chat)
            
            # Display the result with conditional formatting
            if sentiment == "1":
                st.markdown("<h3 style='color: green;'>😊 Positive Sentiment</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h3 style='color: red;'>😔 Negative Sentiment</h3>", unsafe_allow_html=True)
        else:
            # Handle case when no input is provided
            st.warning("Please Enter Some Text to Analyze.")

# Add footer
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: black;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
    <p>Made with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
