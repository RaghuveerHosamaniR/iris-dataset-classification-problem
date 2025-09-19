# ==========================================================
# Iris Dataset Classification App (Streamlit)
# ==========================================================
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IRIS.csv")
    return df

df = load_data()

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Iris ML Project 🌸")
page = st.sidebar.radio("Go to", ["Dataset", "Visualization", "Model Training", "Prediction", "Comparison"])

# -----------------------------
# Dataset Page
# -----------------------------
if page == "Dataset":
    st.title("📊 Iris Dataset")
    st.write("This is the famous Iris dataset with 3 flower species.")

    st.dataframe(df.head())

    st.write("### Dataset Info")
    st.write(df.describe())

    st.write("### Data Types")
    st.write(df.dtypes)

# -----------------------------
# Visualization Page
# -----------------------------
elif page == "Visualization":
    st.title("📈 Data Visualization")

    st.write("### Pairplot of Features")
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    le = LabelEncoder()
    df["species_encoded"] = le.fit_transform(df["species"])
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(plt)

# -----------------------------
# Model Training Page
# -----------------------------
elif page == "Model Training":
    st.title("🤖 Train Models on Iris Dataset")

    # Features and Target
    X = df.drop(columns=["species"])
    y = LabelEncoder().fit_transform(df["species"])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Select model
    model_choice = st.selectbox(
        "Choose a Model",
        ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]
    )

    # Train model
    if st.button("Train Model"):
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=200)
        elif model_choice == "Decision Tree":
            model = DecisionTreeClassifier()
        elif model_choice == "Random Forest":
            model = RandomForestClassifier()
        elif model_choice == "SVM":
            model = SVC()
        elif model_choice == "KNN":
            model = KNeighborsClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ {model_choice} Accuracy: {acc:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        st.write("### Confusion Matrix")
        st.write(cm)

        # Classification Report
        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))

# -----------------------------
# Prediction Page
# -----------------------------
elif page == "Prediction":
    st.title("🌼 Predict Iris Flower Species")

    st.write("Enter the flower measurements:")

    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sw = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    pw = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2)

    # Model for prediction
    model = RandomForestClassifier()
    X = df.drop(columns=["species"])
    y = LabelEncoder().fit_transform(df["species"])
    model.fit(X, y)

    if st.button("Predict"):
        prediction = model.predict([[sl, sw, pl, pw]])[0]
        species = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"][prediction]
        st.success(f"🌸 Predicted Species: **{species}**")
# -----------------------------
# Comparison Page
# -----------------------------
elif page == "Comparison":
    st.title("📊 Model Accuracy Comparison")

    # Features and Target
    X = df.drop(columns=["species"])
    y = LabelEncoder().fit_transform(df["species"])

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=3, min_samples_split=5),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        results.append({"Model": name, "Train Accuracy": train_acc, "Test Accuracy": test_acc})

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    st.write("### Final Accuracy Comparison")
    st.dataframe(results_df)

    # Plot Train vs Test Accuracy
    st.write("### Train vs Test Accuracy Bar Chart")
    results_melt = results_df.melt(id_vars="Model", value_vars=["Train Accuracy", "Test Accuracy"],
                                   var_name="Metric", value_name="Accuracy")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=results_melt, x="Model", y="Accuracy", hue="Metric")
    plt.ylim(0.8, 1.05)
    plt.xticks(rotation=30)
    st.pyplot(plt)
