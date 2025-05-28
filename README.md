# Movie-Box-Office-Revenue-Prediction
**✅ Objective**

  1.To build a machine learning-based system that predicts the potential box office revenue of a movie.
  
  2.To provide reverse suggestions for budget and runtime based on a target revenue.
  
  3.To develop an interactive and user-friendly web interface using Streamlit.

  
**📊 Key Features**

  1.Predicts revenue using two regression models: Random Forest and XGBoost.
  
  2.Accepts user inputs like budget, runtime, popularity, vote average, and genre.
  
  3.Reverse prediction feature: suggests budget and runtime for a user-defined revenue target.
  
  4.Web application built using Streamlit for real-time predictions and visualization.

  
**🧠 Machine Learning Approach**

  1.Random Forest Regressor: Ensemble of decision trees to reduce variance and improve accuracy.
  
  2.XGBoost Regressor: Advanced boosting algorithm optimized for performance and speed.

  
**🧹 Data Preprocessing**

  1.Source Dataset: movies_metadata.csv from TMDb via Kaggle.
  
  2.Steps involved:
  
      1.Dropped rows with missing critical values (budget, revenue, etc.).
      
      2.Converted features like budget, runtime, popularity to numeric.
      
      3.Applied log transformation to budget and revenue to normalize skewed data.
      
      4.Encoded genres using LabelEncoder.
      
      5.Scaled numerical features using StandardScaler.

      
**🧪 Model Evaluation**

  1.Train-Test Split: 80-20
  
  2.Evaluation Metric: Root Mean Squared Error (RMSE)
  
  3.Predicted revenue values are exponentially transformed to return to actual scale from log.


**🖥️ Web Application (Streamlit)**

  1.Interactive form to input movie features.
  
  2.Real-time predictions shown for both Random Forest and XGBoost.
  
  3.Revenue predictions displayed as metrics and in a comparison table.
  
  4.Reverse input section allows the user to enter a target revenue and receive suggested feature values.
  

**🔁 Reverse Prediction Functionality**

  1.Accepts target revenue and genre.
  
  2.Internally estimates budget (e.g., 30% of revenue), fixed popularity and rating.
  
  3.Standardizes inputs before passing them to the model pipeline.


**🔧 Tech Stack**

  1.Python 3.10+
  
  2.Pandas, NumPy – Data cleaning and manipulation.
  
  3.Scikit-learn – Preprocessing, Random Forest model.
  
  4.XGBoost – Advanced regression modeling.
  
  5.Streamlit – UI development and web app deployment.
  
  6.Matplotlib, Seaborn – Visualization (for analysis phase).
  

**📈 Results**

  1.Both models performed well, with XGBoost slightly outperforming Random Forest in RMSE.
  
  2.The system can predict revenue for a wide range of feature combinations with reasonable accuracy.
  
  3.Reverse prediction is a novel, practical feature for strategic movie planning.
  

**🧭 System Architecture**

  1.User Input (via Streamlit UI)
  
  2.Preprocessing (scaling, encoding, transformation)
  
  3.Model Prediction (Random Forest/XGBoost)
  
  4.Revenue Output (log-inverse transformed)
  
  5.Optional Reverse Suggestion (based on target revenue)
  

**🌱 Future Enhancements**

  1.Add features like cast popularity, director ratings, release month, and marketing spend.
  
  2.Apply NLP techniques to analyze movie scripts and reviews.
  
  3.Expand into a full-fledged movie recommendation engine using collaborative filtering.
  
  4.Host the web app on cloud platforms like Heroku or Streamlit Cloud.
