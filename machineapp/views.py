
from django.shortcuts import render, redirect
import pandas as pd
import matplotlib.pyplot as plt
import io, base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from mpl_toolkits.mplot3d import Axes3D 

def home(request):
    return render(request, "machineapp/home.html")
# ----------------------------------------------------------------------------------------------------------------------------------------------

def file_upload(request):
    context = {}

    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]

        df = pd.read_csv(csv_file)
        df = df.dropna()
        table_html = df.to_html(classes="table table-striped", index=False)
        required_cols = ["Temperature", "Vibration", "Pressure", "Failure"]
        for col in required_cols:
            if col not in df.columns:
                context["error"] = f"Column '{col}' missing in CSV!"
                return render(request, "machineapp/upload_file.html", context)
        request.session["uploaded_df"] = df.to_json()
        context["table"] = table_html

        return render(request, "machineapp/upload_file.html", context)

    return render(request, "machineapp/upload_file.html", context)

# ------------------------------------------------------------------------------------------------------------------------------------------
def predict(request):
    if "uploaded_df" not in request.session:
        return render(request, "machineapp/result.html", {"error": "No data uploaded!"})

    df = pd.read_json(request.session["uploaded_df"])
    X = df[["Temperature", "Vibration", "Pressure"]]
    y = df["Failure"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    df["Predicted_Class"] = model.predict(X)
    df["Failure_Probability"] = model.predict_proba(X)[:, 1] * 100
    df["Failure_Probability"] = df["Failure_Probability"].round(2)
    predicted_value = round(df["Failure_Probability"].mean(), 2)
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    context = {
        "predicted_value": predicted_value,
        "accuracy": acc,
        "confusion_matrix": cm.tolist()
    }
    graph_type = request.GET.get("graph_type", "Correlation Heatmap")
    graph_img = None

    def get_graph():
        plt.switch_backend('AGG')
        fig = plt.figure(figsize=(6, 4))

        if graph_type == "Correlation Heatmap":
            corr = df.corr()
            plt.imshow(corr, cmap="coolwarm")
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
            plt.yticks(range(len(corr.columns)), corr.columns)
            plt.title("Correlation Heatmap")

        elif graph_type == "Failure Probability Histogram":
            df["Failure_Probability"].hist()
            plt.title("Failure Probability Distribution (0-100%)")
            plt.xlabel("Probability (%)")
            plt.ylabel("Count")

        elif graph_type == "Temperature vs Failure Probability":
            plt.scatter(df["Temperature"], df["Failure_Probability"])
            plt.xlabel("Temperature")
            plt.ylabel("Failure Probability (%)")
            plt.title("Temperature vs Failure Probability")

        elif graph_type == "Vibration vs Failure Probability":
            plt.scatter(df["Vibration"], df["Failure_Probability"])
            plt.xlabel("Vibration")
            plt.ylabel("Failure Probability (%)")
            plt.title("Vibration vs Failure Probability")

        elif graph_type == "Pressure vs Failure Probability":
            plt.scatter(df["Pressure"], df["Failure_Probability"])
            plt.xlabel("Pressure")
            plt.ylabel("Failure Probability (%)")
            plt.title("Pressure vs Failure Probability")

        elif graph_type == "Pie Chart (Predicted Class)":
            pc = df["Predicted_Class"].value_counts()
            plt.pie(pc, labels=["Healthy (0)", "Fail (1)"], autopct="%1.1f%%")
            plt.title("Predicted Class Pie Chart")

        elif graph_type == "3D Scatter (All Features)":
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df["Temperature"], df["Pressure"], df["Vibration"])
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Pressure")
            ax.set_zlabel("Vibration")
            ax.set_title("3D Scatter Plot")

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches='tight')
        buffer.seek(0)
        graph = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return graph

    graph_img = get_graph()
    context["graph"] = graph_img
    context["graph_type"] = graph_type
    context["graph_options"] = [
        "Correlation Heatmap",
        "Failure Probability Histogram",
        "Temperature vs Failure Probability",
        "Vibration vs Failure Probability",
        "Pressure vs Failure Probability",
        "Pie Chart (Predicted Class)",
        "3D Scatter (All Features)",
    ]
    return render(request, "machineapp/result.html", context)











