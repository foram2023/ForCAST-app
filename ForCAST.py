# Created by Rohan Subba
# Date: May 2025

# This work is a part of a research publication.
#Title : "Machine Learning Approach to Pore-Proxy in Benthic Foraminifera for Paleo-Reconstruction of Dissolved Oxygen in a Lagoon"

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import io

st.set_page_config(
    page_title="֎ ForCAST",
    layout="centered",  
    initial_sidebar_state="expanded"
)

#Main
st.markdown("<h1 style='text-align: center;'>֎ ForCAST</h1>", unsafe_allow_html=True)
st.caption("<p style='font-size:18px; text-align: center; font-weight:normal;'> \"The purpose of computing is insight, not numbers.\" - <b>Richard Hamming</b> </p>", unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = "Random Forest"  

#Sidebar
with st.sidebar:
    st.markdown("<p style='font-size:25px; font-weight:bold;'>֎ ForCAST</p>", unsafe_allow_html=True)

    with st.expander("**📂 Upload File**", expanded=False):  # Upload file
        st.markdown("<div style='font-size:16px; font-weight:bold;'>Upload your data</div>", unsafe_allow_html=True)
        st.session_state.data = st.file_uploader(
            "Upload File", 
            type=["csv", "xlsx"], 
            key="sidebar_uploader",
            label_visibility="collapsed"
        )
        
    # Choose parameters
    st.sidebar.markdown("<div style='font-size:18px; font-weight:bold;'>⚙️ Set Parameters</div>", unsafe_allow_html=True)
    
    # Optimize model parameters
    st.session_state.model_type = st.radio(
        "Select Model:",
        ["Random Forest", "SVR"], 
        key="model_radio"
    )

    if st.session_state.model_type == "Random Forest": # slider parameters for RF 
        st.session_state.rf_test_size = st.slider("Test Size (%)", 0.1, 0.5, 0.2, step=0.05)

        rf_n_estimators = st.slider("n_estimators (Default = 100)", 0, 1000, 100, step=5)
        st.session_state.rf_n_estimators = 100 if rf_n_estimators == 0 else rf_n_estimators


        rf_max_depth = st.slider("Max Depth (0 = None)", 0, 200, 20, step=1)
        st.session_state.rf_max_depth = None if rf_max_depth == 0 else rf_max_depth

        st.session_state.rf_min_samples_split = st.slider("Min Samples Split", 2, 100, 10, step=1)

        st.session_state.rf_min_samples_leaf = st.slider("Min Samples Leaf", 1, 50, 5, step=1)


    elif st.session_state.model_type == "SVR":
        st.session_state.svr_test_size = st.slider("Test Size (%)", 0.1, 0.5, 0.2, step=0.05)
        st.session_state.svr_kernel = st.selectbox("Kernel Type", ["linear", "poly", "rbf", "sigmoid"], index=2)
        st.session_state.svr_c = st.slider("C (Regularization)", 0.01, 10.0, 1.0, step=0.01)
        st.session_state.svr_epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1, step=0.01)

#Load Data
if st.session_state.data is not None:
    if st.session_state.data.name.endswith('.csv'):
        st.session_state.df = pd.read_csv(st.session_state.data)
    elif st.session_state.data.name.endswith('.xlsx'):
        st.session_state.df = pd.read_excel(st.session_state.data)

df = st.session_state.df 

st.markdown("""
    <style>
        /* Style the Tabs */
        div.stTabs button {
            font-size: 20px !important;  /* Increase font size */
            font-weight: bold !important; /* Bold text */
            color: #333333 !important;  /* Dark Gray Text */
            background-color: #f0f0f5 !important; /* Light Gray Background */
            border-radius: 10px 10px 0px 0px !important; /* Rounded Top */
            padding: 1.5% 2% !important;  /* Adjust padding to fit window size */
            border: 1px solid #d1d1e0 !important; /* Light border */
        }
        /* Hover Effect */
        div.stTabs button:hover {
            background-color: #e0e0eb !important; /* Lighter Gray on Hover */
        }
        /* Selected Tab */
        div.stTabs button[selected] {
            background-color: #ccccff !important; /* Soft Blue */
            color: #000000 !important;  /* Black text */
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

#Tab layout
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  ℹ️  About", 
    "📶 DATA VISUALIZATION", 
    "⚛️ RANDOM FOREST", 
    "✳️ SVR", 
    "💻 Developer"
])

#Tab 1
with tab1:
    st.markdown(
        """
        <div style="text-align: justify; padding: 15px; border-radius: 10px; background-color: #f0f2f6; border-left: 6px solid #2196F3;">
            <p style="font-size: 24px; font-weight: bold; font-style: italic; color: #0D47A1;">Welcome to ForCAST!</p>
            <p style="font-size: 18px;">
                This application provides an interactive platform for training, evaluating, and visualizing machine learning models, 
                specifically designed for regression tasks. It supports advanced algorithms like 
                <b>Random Forest Regression(RF)</b> and <b>Support Vector Regression (SVR)</b> to deliver accurate predictions and insightful data analysis.
            </p>
            <ul style="text-align: left; display: inline-block; font-size: 17px; padding-left: 20px; line-height: 1.6;">
                <li><i>Upload your dataset in CSV or Excel format.</i></li>
                <li><i>Analyze correlations between dataset features.</i></li>
                <li><i>Visualize and select key features for your model.</i></li>
                <li><i>Train machine learning models using <b>RFR</b> and <b>SVR</b>.</i></li>
                <li><i>Generate predictions based on your trained models.</i></li>
                <li><i>Download the predicted data as an Excel file.</i></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

#Tab 2
with tab2:
    st.markdown("""
    <style>
        /* Keep the Tab Bar Fixed */
        div.stTabs [role="tablist"] {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 50px !important;  /* Fix tab height */
        }

        /* Rounded Box Styling */
        .custom-box {
            background-color: #e6f2ff;  /* Light Blue Background */
            padding: 6px 10px;  /* Padding inside the box */
            border-radius: 12px;  /* Rounded Corners */
            border: 1px solid #b3d9ff;  /* Light Blue Border */
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.1);  /* Subtle Shadow */
            font-size: 18px;  /* Font Size */
            font-weight: normal;  /* Bold Text */
            color: #000000 !important;  /* Black Text */
            display: inline-block;  /* Ensures the box wraps around text */
            margin: 10px 0;  /* Spacing around boxes */
            text-align: center;  /* Center text */
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='custom-box'>Data Preview</div>", unsafe_allow_html=True)

    data_placeholder = st.empty()

    if df is not None:

        data_placeholder.dataframe(df, height=400, use_container_width=True)
    else:
        data_placeholder.markdown(
            "<div style='height:400px; border:1px dashed #ccc; text-align:center; line-height:400px;'>"
            "Please upload your file to preview data."
            "</div>", 
            unsafe_allow_html=True
        )
    if df is not None:
        #Create correlation map
        st.markdown("<div class='custom-box'>Correlation Map</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))  # Optimized figure size
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

        # Correlation pairs
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        top_5_pairs = (
            upper_tri.unstack()
            .sort_values(ascending=False)
            .dropna()
            .head(5)
        )
        top_5_df = pd.DataFrame(top_5_pairs, columns=["Correlation"]).reset_index()
        top_5_df.columns = ["Feature 1", "Feature 2", "Correlation"]
        top_5_df.index = np.arange(1, len(top_5_df) + 1)
        st.markdown("<div class='custom-box'>Best Correlations</div>", unsafe_allow_html=True)
        st.table(top_5_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('text-align', 'center')]},
             {'selector': 'td', 'props': [('text-align', 'center')]}]
    ))
    else:
        st.warning("⚠️ Please upload your file from the sidebar.")

if 'rf_model' not in st.session_state:
    st.session_state.rf_model = None
if 'svr_model' not in st.session_state:
    st.session_state.svr_model = None

st.markdown(
    """
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)
with tab3:
    st.info("👉 Select the desired model from the sidebar to fine-tune its parameters and optimize performance.")
    if df is not None:
        st.markdown("<div class='custom-box'>Select Variables</div>", unsafe_allow_html=True) # Select input and output features
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("##### ⬇️ Select Input:")
            feature_names = st.multiselect(
                "",
                options=df.columns.tolist(),
                key='features'
            )

        with col2:
            st.markdown("##### ⬆️ Select Output:")
            target_name = st.selectbox(
                "",
                options=df.columns.tolist(),
                key='target'
            )

        if not feature_names or not target_name:
            st.error("⚠️ Please select both input and output features.")
        elif target_name in feature_names:
            st.error("⚠️ Output feature cannot be one of the input feature.")
        else:
            X = df[feature_names]
            y = df[target_name]

            if st.session_state.model_type == "Random Forest":
                test_size = st.session_state.rf_test_size
                n_estimators = st.session_state.rf_n_estimators
                max_depth = st.session_state.rf_max_depth
                min_samples_split = st.session_state.rf_min_samples_split
                min_samples_leaf = st.session_state.rf_min_samples_leaf

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )

                #Train RF Model
                rf_model = RandomForestRegressor(
                    n_estimators=n_estimators if n_estimators is not None else 100,  # Default = 100
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42
                )
                rf_model.fit(X_train, y_train)
                st.session_state.rf_model = rf_model
                y_pred = rf_model.predict(X_test)

                # Feature importance
                feature_importance = rf_model.feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importance
                }).sort_values(by='Importance', ascending=False)
                
                st.markdown("<div class='custom-box'> Feature Importance</div>", unsafe_allow_html=True)  
                fig_pie = px.pie(
                    importance_df,
                    values='Importance',
                    names='Feature'
                )
                fig_pie.update_layout(
                    height=350, width=350,
                    legend=dict(
                        orientation="h",
                        title=None,
                        yanchor="top",
                        y=-0.2,  
                        xanchor="center",
                        x=0.5,
                        font=dict(size=14),  
                        itemsizing="constant"
                    ),
                    margin=dict(l=10, r=10, t=10, b=10)
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                #Performance evaluation
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                # Performance metrics
                st.markdown("<div class='custom-box' style='text-align: center;'> Performance Metrics</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <style>
                        .metric-table {{
                            width: 50%;
                            margin-left: auto;
                            margin-right: auto;
                            text-align: center;
                            border-collapse: collapse;
                            font-size: 16px;
                        }}
                        .metric-table th, .metric-table td {{
                            border: 1.5px solid #ddd;
                            padding: 6px;
                        }}
                        .metric-table th {{
                            font-weight: bold;
                            color: black;
                            background-color: white;
                        }}
                        .metric-table td {{
                            font-weight: normal;
                            color: black;
                        }}
                        .metric-table td.metric-value {{
                            color: #39db44;
                            font-weight: bold;
                            font-size: 18px;
                        }}
                    </style>
                    <table class="metric-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Mean Squared Error (MSE)</td>
                            <td class="metric-value">{mse:.3f}</td>
                        </tr>
                        <tr>
                            <td>Root Mean Squared Error (RMSE)</td>
                            <td class="metric-value">{rmse:.3f}</td>
                        </tr>
                        <tr>
                            <td>Mean Absolute Error (MAE)</td>
                            <td class="metric-value">{mae:.3f}</td>
                        </tr>
                        <tr>
                            <td>R² Score</td>
                            <td class="metric-value">{r2:.3f}</td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )

                #Prediction
                st.markdown("<div class='custom-box'> Make Predictions</div>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:18px; font-style:italic;'>Enter values for the input variables in the table below:</p>", unsafe_allow_html=True)

                input_df = pd.DataFrame(columns=feature_names)

                #Table
                input_df = st.data_editor(
                    input_df.reset_index(drop=True),
                    num_rows="dynamic",
                    key="prediction_input_table"
                )

                # Create predictions
                if st.button("🕹️ Predict", key="predict_button", help="Click to make predictions"):
                    if input_df.empty:
                        st.warning("⚠️ Please enter values before predicting.")
                    else:
                        predictions = rf_model.predict(input_df)  
                        input_df["Prediction"] = predictions  # Add predictions

                        input_df.index = range(1, len(input_df) + 1)  

                        st.markdown("<div class='custom-box'> Predicted Results</div>", unsafe_allow_html=True)
                        st.table(input_df)  

                        # option to download predicted data
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                            input_df.to_excel(writer, sheet_name="Predictions", index=True)  # Save to Excel
                            writer.close()

                        # Download button
                        st.download_button(
                            label="📥 Download",
                            data=excel_buffer.getvalue(),
                            file_name="predictions.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
    else:
        st.warning("⚠️ Please upload your file from the sidebar.")


with tab4:
    st.info("👉 Select the desired model from the sidebar to fine-tune its parameters and optimize performance.")
    if df is not None:
        st.markdown("<div class='custom-box'>Select Variables</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("##### ⬇️ Select Input:")
            svr_feature_names = st.multiselect(
                "",
                options=df.columns.tolist(),
                key='svr_features'
            )

        with col2:
            st.markdown("##### ⬆️ Select Output:")
            svr_target_name = st.selectbox(
                "",
                options=df.columns.tolist(),
                key='svr_target'
            )

        if 'svr_test_size' not in st.session_state:
            st.session_state.svr_test_size = 0.2
        if 'svr_kernel' not in st.session_state:
            st.session_state.svr_kernel = 'rbf'
        if 'svr_c' not in st.session_state:
            st.session_state.svr_c = 1.0
        if 'svr_epsilon' not in st.session_state:
            st.session_state.svr_epsilon = 0.1

        if not svr_feature_names or not svr_target_name:
            st.error("⚠️ Please select both input and output features.")
        elif svr_target_name in svr_feature_names:
            st.error("⚠️ Output feature cannot be one of the input feature.")
        else:
            X = df[svr_feature_names]
            y = df[svr_target_name]

            test_size = st.session_state.svr_test_size
            kernel = st.session_state.svr_kernel
            C = st.session_state.svr_c
            epsilon = st.session_state.svr_epsilon

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Train SVR Model
            svr_model = SVR(kernel=kernel, C=C, epsilon=epsilon)
            svr_model.fit(X_train, y_train)
            st.session_state.svr_model = svr_model  # Save model in session state
            y_pred = svr_model.predict(X_test)

            # 🔹 Performance metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.markdown("<div class='custom-box' style='text-align: center;'> Performance Metrics</div>", unsafe_allow_html=True)
            st.markdown(
                    f"""
                    <style>
                        .metric-table {{
                            width: 50%;
                            margin-left: auto;
                            margin-right: auto;
                            text-align: center;
                            border-collapse: collapse;
                            font-size: 16px;
                        }}
                        .metric-table th, .metric-table td {{
                            border: 1.5px solid #ddd;
                            padding: 6px;
                        }}
                        .metric-table th {{
                            font-weight: bold;
                            color: black;
                            background-color: white;
                        }}
                        .metric-table td {{
                            font-weight: normal;
                            color: black;
                        }}
                        .metric-table td.metric-value {{
                            color: #39db44;
                            font-weight: bold;
                            font-size: 18px;
                        }}
                    </style>
                    <table class="metric-table">
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Mean Squared Error (MSE)</td>
                            <td class="metric-value">{mse:.3f}</td>
                        </tr>
                        <tr>
                            <td>Root Mean Squared Error (RMSE)</td>
                            <td class="metric-value">{rmse:.3f}</td>
                        </tr>
                        <tr>
                            <td>Mean Absolute Error (MAE)</td>
                            <td class="metric-value">{mae:.3f}</td>
                        </tr>
                        <tr>
                            <td>R² Score</td>
                            <td class="metric-value">{r2:.3f}</td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )

            # Prediction
            st.markdown("<div class='custom-box'> Make Predictions</div>", unsafe_allow_html=True)
            st.markdown("<p style='font-size:18px; font-style:italic;'>Enter values for the input variables in the table below:</p>", unsafe_allow_html=True)

            input_df = pd.DataFrame(columns=svr_feature_names)
            input_df = st.data_editor(
                input_df.reset_index(drop=True),
                num_rows="dynamic",
                key="svr_prediction_input_table"
            )

            if st.button("🔮 Predict", key="svr_predict_button", help="Click to make predictions"):
                if input_df.empty:
                    st.warning("⚠️ Please enter values before predicting.")
                else:
                    predictions = svr_model.predict(input_df)
                    input_df["Prediction"] = predictions
                    input_df.index = range(1, len(input_df) + 1)

                    st.markdown("<div class='custom-box'> Predicted Results</div>", unsafe_allow_html=True)
                    st.table(input_df)

                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                        input_df.to_excel(writer, sheet_name="Predictions", index=True)
                        writer.close()

                    st.download_button(
                        label="📥 Download",
                        data=excel_buffer.getvalue(),
                        file_name="svr_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    else:            
        st.warning("⚠️ Please upload your file from the sidebar.")

with tab5:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px; border-radius: 10px; background-color: #f0f2f6; border-left: 6px solid #2196F3;">
            <p style="font-size: 16px; font-weight: normal;">֎ ForCAST is developed by Rohan Subba (INDIA).</p>
            <p style="font-size: 15px;">✉️ For any questions or feedback, feel free to reach out via email:</p>
            <p style="font-size: 16px; font-weight: bold; color: blue;">
                <a href="mailto: subbaforams@gmail.com" style="text-decoration: none; color: blue;">
                    subbaforams@gmail.com
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

