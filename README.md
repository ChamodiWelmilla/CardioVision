# CardioVision 🫀

CardioVision is an AI-powered cardiovascular risk analysis application. It utilizes unsupervised machine learning clustering models (K-Means, DBSCAN, and GMM) to group cardiovascular profiles and assess potential health risks based on patient vitals. 

This project features a complete end-to-end pipeline: from data exploration in Jupyter Notebooks to a production-ready Flask Web API, all tied together with a stunning, premium web interface.

## 🚀 Features
- **AI Clustering Analysis**: Uses K-Means to cluster patients into distinct cardiovascular risk profiles based on historical data.
- **Interactive UI**: A dynamic, premium user interface built with HTML, Vanilla CSS, and modern aesthetics (glassmorphism, gradient text, and micro-animations).
- **Automated Pipeline**: Includes scripts to automate environment setup, model training, artifact serialization (Joblib), and server hosting.

## 📂 Project Structure

- `Cardio_Vision_Clustering.ipynb` - The original Jupyter Notebook used for Exploratory Data Analysis (EDA) and evaluating different clustering algorithms (K-Means, DBSCAN, Gaussian Mixture Models).
- `heart_diseases.csv` - The cardiovascular dataset used for training the model.
- `train_model.py` - A Python script that preprocesses the dataset, trains the K-Means model, and saves the trained model (`kmeans_model.pkl`) and scaler (`scaler.pkl`) for production use.
- `app.py` - The Flask web server backend that loads the trained artifacts, exposes a `/predict` API endpoint, and serves the UI.
- `templates/index.html` - The frontend user interface.
- `start.bat` - A convenient Windows startup script that installs dependencies, trains the model, and launches the Flask server automatically.

## 🛠️ How to Run Locally

### Using the Automated Startup Script (Windows)
1. Open your File Explorer or Terminal in the project directory.
2. Run the `start.bat` script:
   ```cmd
   .\start.bat
   ```
   *(This script will automatically install missing Python dependencies, run `train_model.py` to generate the `.pkl` files, and boot up the Flask server).*
3. Open your web browser and navigate to **[http://localhost:5000](http://localhost:5000)**.

### Manual Setup (Cross-Platform)
1. Install the required dependencies:
   ```bash
   pip install flask scikit-learn pandas joblib
   ```
2. Train the model and generate the scaler and model artifacts:
   ```bash
   python train_model.py
   ```
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Access the application at **http://localhost:5000** in your browser.

## 📊 Vitals Explained
The AI model categorizes users based on the following input parameters:
- **Age**: Patient's age in years.
- **Sex**: Patient's sex (Male/Female).
- **Resting Blood Pressure**: Measured in mm Hg.
- **Cholesterol**: Serum cholesterol in mg/dl.
- **Max Heart Rate**: Maximum heart rate achieved during observation.

