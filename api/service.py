import numpy as np
import bentoml
from bentoml.io import NumpyNdarray, Image
from PIL.Image import Image as PILImage
from PIL import Image
from fastapi import FastAPI
app = FastAPI()

bento_model_2 = bentoml.mlflow.import_model(
    "mlflow_pytorch",
    "models:/conv1/1",
     signatures={"predict": {"batchable": True}},
 )
mnist_runner = bentoml.mlflow.get("mlflow_pytorch:latest").to_runner()

svc = bentoml.Service("pytorch_mnist", runners=[mnist_runner])
svc.mount_asgi_app(FastAPI)
@app.post("/predict")
def predict():
    image = Image.open("api/img_4.jpg")
    img_arr = np.array(image)/255.0
    input_arr = np.expand_dims(img_arr, 0).astype("float32")
    output_tensor = mnist_runner.predict.run(input_arr)
    return output_tensor.numpy()