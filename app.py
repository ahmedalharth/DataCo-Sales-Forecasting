import streamlit as st
import pickle
import pandas as pd
import imblearn
import lightgbm as lgb
from sklearn.preprocessing import  LabelEncoder  , StandardScaler


numerical_col = [
    'order_item_discount_rate',
    'order_item_product_price',
    'order_item_quantity'
]
categorical_col = [
    'type', 'delivery_status', 'customer_country', 
    'customer_segment', 'market', 'shipping_mode'
]


# Define the label encoder function
def lable_encoder(df):
    for col in categorical_col:  # Ensure 'cat' is passed or globally defined
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df


# Load the saved pipeline
@st.cache_resource
def load_pipeline():
    with open('Pipeline/lgb_pipeline.pkl', 'rb') as f:
        return pickle.load(f)

# Load the pipeline
pipeline = load_pipeline()

# Streamlit app UI
st.title("LightGBM Prediction App")
st.write("Enter input features to get predictions using the LightGBM model.")





# Application
########################################################################################################################
# Streamlit App
st.title("Order Prediction App")
st.write("Provide details about the order to predict outcomes.")

# Feature input sections
with st.expander("Customer Information", expanded=True):
    customer_country = st.text_input("Customer Country", value="USA")
    customer_segment = st.selectbox("Customer Segment", options=['Consumer', 'Corporate', 'Home Office'])

with st.expander("Order Information", expanded=True):
    type_ = st.selectbox("Product Type", options=['Type1', 'Type2', 'Type3'])
    delivery_status = st.selectbox("Delivery Status", options=['Delivered', 'Pending', 'In Transit'])
    market = st.selectbox("Market", options=['US', 'EU', 'APAC', 'MEA', 'LATAM'])
    shipping_mode = st.selectbox("Shipping Mode", options=['First Class', 'Second Class', 'Standard Class'])
    month_order_date = st.slider("Month of Order Date", min_value=1, max_value=12, step=1)
    year_order_date = st.slider("Year of Order Date", min_value=2000, max_value=2030, step=1)

with st.expander("Product and Discount Information", expanded=True):
    order_item_quantity = st.number_input("Order Item Quantity", min_value=1, value=1)
    order_item_discount_rate = st.slider("Order Item Discount Rate", min_value=0.0, max_value=1.0, step=0.01)
    order_item_product_price = st.number_input("Order Item Product Price", min_value=0.0, value=100.0)
    discount_per_product = st.number_input("Discount Per Product", min_value=0.0, value=0.0)
    benefit_per_product = st.number_input("Benefit Per Product", min_value=0.0, value=0.0)
    total_discount_per_product = st.number_input("Total Discount Per Product", min_value=0.0, value=0.0)
    max_discount_per_order = st.number_input("Max Discount Per Order", min_value=0.0, value=0.0)
    product_name_mean = st.number_input("Product Name Mean", min_value=0.0, value=0.0)

# Combine inputs into a DataFrame
input_data = pd.DataFrame({
    'type': [type_],
    'delivery_status': [delivery_status],
    'customer_country': [customer_country],
    'customer_segment': [customer_segment],
    'market': [market],
    'shipping_mode': [shipping_mode],
    'order_item_discount_rate': [order_item_discount_rate],
    'order_item_product_price': [order_item_product_price],
    'order_item_quantity': [order_item_quantity],
    'Month_order_date_(dateorders)': [month_order_date],
    'Year_order_date_(dateorders)': [year_order_date],
    'DiscountPerProduct': [discount_per_product],
    'DenefitPerProduct': [benefit_per_product],
    'TotalDiscountPerProduct': [total_discount_per_product],
    'MaxDiscountPerOrder': [max_discount_per_order],
    'product_name_mean': [product_name_mean]
})

# Display user inputs
st.subheader("Input Data")
st.write(input_data)

# Prediction
if st.button("Predict"):
    try:
        prediction = pipeline.predict(input_data)
        st.success(f"The predicted outcome is: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Batch Predictions
st.subheader("Batch Processing")
st.write("Upload a CSV file with the same structure as the input data to predict outcomes for multiple records.")
uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
if uploaded_file is not None:
    try:
        batch_data = pd.read_csv(uploaded_file)
        predictions = pipeline.predict(batch_data)
        st.write("Predictions:")
        st.write(predictions)
    except Exception as e:
        st.error(f"An error occurred: {e}")
########################################################################################################################
