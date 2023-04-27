import os
import matplotlib.pyplot as plt
from mlflow.deployments import get_deploy_client
from PIL import Image
import numpy as np
from torchvision import transforms



def predict():
    plugin = get_deploy_client("torchserve")
    image = Image.open("api/img_4.jpg")
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    image_tensor = mnist_transforms(image)
    #img = np.array(image, dtype=np.float32).flatten()[np.newaxis, ...]/255
    prediction = plugin.predict("test", image_tensor)
    print("enter")
    print("Prediction Result" + prediction)

if __name__ == "__main__":
    predict()


    