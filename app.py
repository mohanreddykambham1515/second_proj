import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app  = FastAPI(title = 'usa housing price prediction api')

model = joblib.load('house_price_model.pkl')


class HouseFeatures(BaseModel):
    avg_area_income: float
    avg_area_house_age: float
    avg_area_number_of_rooms: float
    avg_area_number_of_bedrooms: float
    area_population: float




@app.get('/')
def home():
    return {'message':'house price prediction api'}

@app.post('/predict')
def predict_price(data:HouseFeatures):
    input_data = pd.DataFrame([{
        "Avg. Area Income": data.avg_area_income,
        "Avg. Area House Age": data.avg_area_house_age,
        "Avg. Area Number of Rooms": data.avg_area_number_of_rooms,
        "Avg. Area Number of Bedrooms": data.avg_area_number_of_bedrooms,
        "Area Population": data.area_population
    
    }])

    prediction = model.predict(input_data)
    return{'prediction_price': float(prediction[0])}