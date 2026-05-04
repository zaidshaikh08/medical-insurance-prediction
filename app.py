from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            logging.info("Received prediction request from web form.")
            
            # Pass the entire HTML form payload to CustomData
            data = CustomData(request.form.to_dict())
            pred_df = data.get_data_as_dataframe()
            
            predict_pipeline = PredictPipeline(tier='C')
            results = predict_pipeline.predict(pred_df)
            
            # Results come back as a NumPy array, extract the first value
            final_cost = round(results[0], 2)
            
            logging.info(f"Prediction successful: ${final_cost}")
            return render_template('index.html', results=final_cost)
            
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return render_template('index.html', error="An error occurred during prediction. Check logs for details.")

if __name__ == "__main__":
    # Run the Flask app on localhost
    app.run(host="0.0.0.0", port=5000, debug=True)