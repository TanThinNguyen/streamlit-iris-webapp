import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Function
@st.cache
def load_dataset():    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y, iris


@st.cache
def create_df(X, y, iris):
    df = pd.DataFrame(
        data=X,
        columns=iris.feature_names
    )
    df["species"] = iris.target_names[y]
    df.rename(lambda x: str(x).replace(" (cm)", ""), axis=1, inplace=True)
    return df


def view_dataset(df):
    # Preview dataset
    st.write("""
    ### Preview dataset
    """)
    if st.button("Head"):
        st.dataframe(df.head(10))

    if st.button("Tail"):
        st.dataframe(df.tail(10))
        
    if st.button("Show all"):
        st.dataframe(df)
        st.write("Rows", len(df.index))
        st.write("Columns", len(df.columns))

    # View df by column
    st.write("""
    ### View columns of dataset
    """)
    col_select = st.multiselect("Select columns to show", df.columns.values)
    if len(col_select) > 0:
        st.dataframe(df[col_select])

    # Summary dataset
    st.write("""
    ### Dataset summary
    """)
    st.write(df.describe())

    fig = px.scatter_matrix(df,
        dimensions=["sepal width", "sepal length", "petal width", "petal length"],
        color="species"
    )
    #st.write(fig)

    st.write("""
    ### Visualize
    """)
    fig = px.scatter(df, x="sepal width", y="sepal length", color="species")
    st.write(fig)

    fig = px.scatter(df, x="petal width", y="petal length", color="species")
    st.write(fig)

    fig = px.box(df, y="sepal width", x="species", color="species")
    st.write(fig)

    fig = px.box(df, y="sepal length", x="species", color="species")
    st.write(fig)

    fig = px.box(df, y="petal width", x="species", color="species")
    st.write(fig)

    fig = px.box(df, y="petal length", x="species", color="species")
    st.write(fig)

    fig = px.histogram(df, x="species", color="species")
    st.write(fig)


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.slider("K", 1, 15, 3)
        params["K"] = K
    elif clf_name == "Random Forest":
        # chiều sâu tối đa của cây
        max_depth = st.slider("Max Depth", 2, 15)
        # số lượng cây
        n_estimators = st.slider("Number of estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimator"] = n_estimators
    return params


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(
            n_neighbors=params["K"],
            weights="distance"
        )
    elif clf_name == "Random Forest":
        clf = RandomForestClassifier(
            n_estimators=params["n_estimator"],
            max_depth=params["max_depth"],
            random_state=1
        )
    return clf


def predict(X, y, iris):
    st.write("""
    ### Classifier
    """)
    clf_name = st.selectbox("Select a classifier", ("KNN", "Random Forest"))

    # Create clf UI
    params = add_parameter_ui(clf_name)

    # Create clf
    clf = get_classifier(clf_name, params)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    clf.fit(X_train, y_train)

    # Accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("""
    ### Accuracy
    """)
    st.write("Accuracy", acc)

    # Input feature
    st.write("""
    ### Input features
    """)
    sepal_len = st.slider("Sepal Length", 1.0, 10.0, 5.0)
    sepal_wid = st.slider("Sepal Width", 1.0, 10.0, 5.0)
    pental_len = st.slider("Pental Length", 1.0, 10.0, 5.0)
    pental_wid = st.slider("Pental Width", 1.0, 10.0, 5.0)
    
    features = np.array([sepal_len, sepal_wid, pental_len, pental_wid])
    features = np.expand_dims(features, axis=0)

    # Predict
    if st.button("Predict"):
        st.write("""
        ### Result
        """)
        labels = clf.predict(features)
        st.write("Species:", iris.target_names[labels[0]])
    


# ---------------------
def main():
    # Title
    st.title("Iris App")

    # Load dataset
    X, y, iris = load_dataset()

    # Create dataframe
    df = create_df(X, y, iris)

    # Sidebar
    action_menu = ["View Dataset", "Prediction"]
    action = st.sidebar.selectbox("Select what you want to do", action_menu)

    if action == "View Dataset":
        #st.subheader("View Dataset")
        view_dataset(df)
    elif action == "Prediction":
        #st.subheader("Prediction")
        predict(X, y, iris)


# ---------------------
if __name__ == "__main__":
    main()