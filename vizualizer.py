import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_data():
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df

def convert_categorical(df, encoding_type):
    categorical_columns = df.select_dtypes(include=['object']).columns

    if encoding_type == 'Label Encoding':
        label_encoder = LabelEncoder()
        for column in categorical_columns:
            df[column] = label_encoder.fit_transform(df[column].astype(str))

    elif encoding_type == 'One-Hot Encoding':
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    return df

def show_scatter_plot(df, x_column, y_column):
    st.subheader("Scatter Plot")
    fig = px.scatter(df, x=x_column, y=y_column, marginal_y="box", marginal_x="histogram")
    st.plotly_chart(fig)

def show_bar_chart(df, x_column, y_column):
    st.subheader("Bar Chart")
    fig = px.bar(df, x=x_column, y=y_column)
    st.plotly_chart(fig)

def show_line_chart(df, x_column, y_column):
    st.subheader("Line Chart")
    fig = px.line(df, x=x_column, y=y_column, title=f'{y_column} over {x_column}')
    st.plotly_chart(fig)

def show_histogram(df, column):
    st.subheader("Histogram")
    fig = px.histogram(df, x=column, title=f'Histogram for {column}')
    st.plotly_chart(fig)

def show_pie_chart(df, column):
    st.subheader("Pie Chart")
    fig = px.pie(df, names=column, title=f'Pie Chart for {column}')
    st.plotly_chart(fig)

def show_box_plot(df, x_column, y_column):
    st.subheader("Box Plot")
    fig = px.box(df, x=x_column, y=y_column, title=f'Box Plot for {y_column} by {x_column}')
    st.plotly_chart(fig)

def show_correlation_heatmap(df):
    st.subheader("Correlation Heatmap")
    corr_matrix = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5, ax=ax)
    st.pyplot(fig)

def perform_kmeans_clustering(df, num_clusters):
    st.subheader("KMeans Clustering")
    features = df.select_dtypes(include=['float64', 'int64'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)

    fig = px.scatter(df, x=features.columns[0], y=features.columns[1], color='Cluster', 
                     title=f'KMeans Clustering ({num_clusters} Clusters)')
    st.plotly_chart(fig)

def show_pair_plot(df):
    st.subheader("Pair Plot")
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    fig = sns.pairplot(df[numerical_columns])
    st.pyplot(fig)

def generate_conclusion(df):
    conclusion = "Automated Conclusion:\n\n"
    
    # General observations
    num_rows, num_cols = df.shape
    conclusion += f"The dataset contains {num_rows} rows and {num_cols} columns.\n"

    # Missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        conclusion += f"There are missing values in the dataset. Consider handling them appropriately.\n"

    # Descriptive statistics
    conclusion += "Descriptive Statistics:\n"
    conclusion += df.describe().to_string() + "\n"

    # Correlation analysis
    correlation_matrix = df.corr()
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

    if high_corr_pairs:
        conclusion += "Highly correlated features found:\n"
        for pair in high_corr_pairs:
            conclusion += f"- {pair[0]} and {pair[1]}\n"

    return conclusion

def main():
    
    st.title("________Visualizer App_____")
    st.subheader("Task 1")
    df = load_data()
    
    if df is not None:
        st.write("Data Preview:")
        st.dataframe(df.head())

        st.sidebar.title("Visualization Options")

        # Encoding options
        encoding_type = st.sidebar.selectbox("Select Encoding Type", ['None', 'Label Encoding', 'One-Hot Encoding'])
        if encoding_type != 'None':
            df = convert_categorical(df, encoding_type)

        x_scatter = st.sidebar.selectbox("Select X-axis column (Scatter Plot)", df.columns)
        y_scatter = st.sidebar.selectbox("Select Y-axis column (Scatter Plot)", df.columns)
        show_scatter_plot(df, x_scatter, y_scatter)

        x_bar = st.sidebar.selectbox("Select X-axis column (Bar Chart)", df.columns)
        y_bar = st.sidebar.selectbox("Select Y-axis column (Bar Chart)", df.columns)
        show_bar_chart(df, x_bar, y_bar)

        x_line = st.sidebar.selectbox("Select X-axis column (Line Chart)", df.columns)
        y_line = st.sidebar.selectbox("Select Y-axis column (Line Chart)", df.columns)
        show_line_chart(df, x_line, y_line)

        histogram_column = st.sidebar.selectbox("Select column for Histogram", df.columns)
        show_histogram(df, histogram_column)

        pie_chart_column = st.sidebar.selectbox("Select column for Pie Chart", df.columns)
        show_pie_chart(df, pie_chart_column)

        x_box = st.sidebar.selectbox("Select X-axis column (Box Plot)", df.columns)
        y_box = st.sidebar.selectbox("Select Y-axis column (Box Plot)", df.columns)
        show_box_plot(df, x_box, y_box)

        if st.sidebar.checkbox("Show Correlation Heatmap"):
            show_correlation_heatmap(df)

        num_clusters = st.sidebar.slider("Select number of clusters (KMeans Clustering)", 2, 10, 3)
        perform_kmeans_clustering(df, num_clusters)

        if st.sidebar.checkbox("Show Pair Plot"):
            show_pair_plot(df)

        # Display conclusions
        conclusion = generate_conclusion(df)
        st.subheader("Automated Analysis and Conclusion")
        st.text_area("Automated Conclusion:", conclusion, height=400)


    st.text("Made By Kunal Sharma")

if __name__ == "__main__":
    main()
