import numpy as np
import bentoml
from PIL import Image
from bentoml.io import NumpyNdarray
mod = bentoml.mlflow.import_model('my_mlflow_model', "models:/conv1/1")
svc = bentoml.Service("conv", runners=[mod])
@svc.api()
def classify(input_series: Image) -> np.ndarray:
    image = Image.open(img)
    # Resize the image to your desired size
    img = np.array(image, dtype=np.float32).flatten()[np.newaxis, ...]/255
    result = mod.predict.run(img)
    return np.array(result)