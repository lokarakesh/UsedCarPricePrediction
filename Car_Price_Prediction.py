{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c1455fb2-1c11-4649-981d-072b695350ee",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Squared Error: 5.6499108521045125\n",
      "Model and label encoders saved successfully!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.metrics import mean_squared_error\n",
    "import joblib\n",
    "\n",
    "# Load the dataset\n",
    "data = pd.read_csv(\"Car_Data_Zscore.csv\")\n",
    "\n",
    "# Identify categorical and numerical columns\n",
    "categorical_columns = ['Make', 'Model', 'Location', 'Fuel_Type', 'Transmission', 'Owner_Type']\n",
    "numeric_columns = ['Year', 'Kilometers_Driven', 'Mileage(KMPL)', 'Engine(CC)', 'Power(BHP)', 'Seats']\n",
    "target_column = 'Price(Lakhs)'\n",
    "\n",
    "# Encode categorical columns\n",
    "label_encoders = {}\n",
    "for col in categorical_columns:\n",
    "    encoder = LabelEncoder()\n",
    "    data[col] = encoder.fit_transform(data[col])\n",
    "    label_encoders[col] = encoder\n",
    "\n",
    "# Split the dataset into features (X) and target (y)\n",
    "X = data[categorical_columns + numeric_columns]\n",
    "y = data[target_column]\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Train a Random Forest Regressor\n",
    "model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Evaluate the model\n",
    "y_pred = model.predict(X_test)\n",
    "mse = mean_squared_error(y_test, y_pred)\n",
    "print(f\"Mean Squared Error: {mse}\")\n",
    "\n",
    "# Save the model and label encoders\n",
    "joblib.dump(model, \"car_price_model.pkl\")\n",
    "joblib.dump(label_encoders, \"label_encoders.pkl\")\n",
    "print(\"Model and label encoders saved successfully!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "f54b0a89-56c8-47e5-8e83-0565d7fe0fad",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (1817081337.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[2], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    streamlit run streamlit_app.py\u001b[0m\n\u001b[1;37m              ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "streamlit run streamlit_app.py"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b17cb921-486b-43fc-a606-c317b5ce6be6",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
