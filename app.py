from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:

        input_data = pd.DataFrame([{

            "OverallQual": int(request.form["OverallQual"]),

            "GrLivArea": float(request.form["GrLivArea"]),

            "GarageCars": int(request.form["GarageCars"]),

            "GarageArea": float(request.form["GarageArea"]),

            "TotalBsmtSF": float(request.form["TotalBsmtSF"]),

            "1stFlrSF": float(request.form["FirstFlrSF"]),

            "FullBath": int(request.form["FullBath"]),

            "TotRmsAbvGrd": int(request.form["TotRmsAbvGrd"]),

            "YearBuilt": int(request.form["YearBuilt"]),

            "YearRemodAdd": int(request.form["YearRemodAdd"])

        }])

        prediction = model.predict(input_data)[0]

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction=f"Error : {e}")


if __name__ == "__main__":
    app.run(debug=True)