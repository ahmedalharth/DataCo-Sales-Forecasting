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





def feature_engineering(df):

    """
    Perform feature engineering for a dataset (training or test).
    This includes calculating delay, discount, and benefit-related features.
    """

    # Delay in orders
    df['DelayOrdered'] = df['days_for_shipment_(scheduled)'] - df['days_for_shipping_(real)']
    df.drop(['days_for_shipment_(scheduled)', 'days_for_shipping_(real)'], axis=1, inplace=True)

    # Discount per product
    discount_map = dict(df.groupby('product_name')['order_item_discount'].max())
    df['DiscountPerProduct'] = df['product_name'].map(discount_map)

    # Benefit per product
    benefit_map = dict(df.groupby('product_name')['benefit_per_order'].mean())
    df['DenefitPerProduct'] = df['product_name'].map(benefit_map)

    # Total discount variance per product
    discount_var_map = dict(df.groupby('product_name')['order_item_discount'].var())
    df['TotalDiscountPerProduct'] = df['product_name'].map(discount_var_map)

    # Max discount per order
    max_discount_map = dict(df.groupby('order_item_id')['order_item_discount'].max())
    df['MaxDiscountPerOrder'] = df['order_item_id'].map(max_discount_map)

    # Drop original columns that are no longer needed
    df.drop(['order_item_discount', 'benefit_per_order', 'order_item_profit_ratio'], axis=1, inplace=True)
    
    return df


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


data_df = pd.DataFrame(
    
    {
        "widgets": ["st.selectbox", "st.number_input", "st.text_area", "st.button"],
        "sales": [

            [0, 4, 26, 80, 100, 40],
            [80, 20, 80, 35, 40, 100],
            [10, 20, 80, 80, 70, 0],
            [10, 100, 20, 100, 30, 100],
        ],
    }
)

st.data_editor(
    data_df,
    column_config={
        "widgets": st.column_config.Column(
            "Streamlit Widgets",
            help="Streamlit **widget** commands 🎈",
            width="medium",
            required=True,
        ),
        "sales": st.column_config.AreaChartColumn(
            "Sales (last 6 months)",
            width="medium",
            help="The sales volume in the last 6 months",
            y_min=0,
            y_max=100,
         ),
    },
    hide_index=True,
    num_rows="dynamic",
)


# Sidebar Input Form for Categorical Features with Descriptions
type_ = st.sidebar.selectbox(
    "Type",
    ["DEBIT", "CREDIT", "CASH", "OTHER"],
    index=0,
    help="Select the payment type. Most common: DEBIT."
)

delivery_status = st.sidebar.selectbox(
    "Delivery Status",
    ["Late delivery", "On time", "Early delivery", "Pending"],
    index=0,
    help="Select the delivery status of the order. Most common: Late delivery."
)

customer_country = st.sidebar.selectbox(
    "Customer Country",
    ["EE. UU.", "Other"],
    index=0,
    help="Select the customer's country. Most common: EE. UU."
)

customer_segment = st.sidebar.selectbox(
    "Customer Segment",
    ["Consumer", "Corporate", "Home Office"],
    index=0,
    help="Select the customer segment. Most common: Consumer."
)

market = st.sidebar.selectbox(
    "Market",
    ["LATAM", "EMEA", "APAC", "USCA", "Other"],
    index=0,
    help="Select the market region. Most common: LATAM."
)

shipping_mode = st.sidebar.selectbox(
    "Shipping Mode",
    ["Standard Class", "First Class", "Second Class", "Other"],
    index=0,
    help="Select the shipping mode. Most common: Standard Class."
)


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
