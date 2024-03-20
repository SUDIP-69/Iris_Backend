import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    slength: float
    swidth: float
    plength: float
    pwidth: float

with open('SVM.pickle', 'rb') as file:
    model = pickle.load(file)

def pred(a, b, c, d):
    dataset = [[a, b, c, d]]
    return model.predict(dataset)[0]

@app.get('/')
def home():
    return {'name':4}
@app.post('/getresult')
def get_result(data: InputData):
    slength = data.slength
    swidth = data.swidth
    plength = data.plength
    pwidth = data.pwidth
    species = pred(slength, swidth, plength, pwidth)

    return {'species': species}

if __name__ == "__main__":
   uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)