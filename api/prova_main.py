import numpy as np
from fastapi import FastAPI
from fastapi import BackgroundTasks
import mlflow
from ml.data import load_cora_dgl, load_cora
from mlflow.tracking import MlflowClient
from api.models import DeleteApiData, TrainApiData, PredictIdApiData, RegisterApiData, TransitionApiData, SaveModelApiData, RegisterSavedModelApiData, RenameModelApiData, PredictNameApiData, RegisterByNameApiData
import torch
from ml.train import Trainer
from PIL import Image
import pickle
import mlflow.pytorch
import requests 
import bentoml
#from unicorn import UnicornMiddleware


app = FastAPI()
mlflowclient = MlflowClient(
    mlflow.get_tracking_uri(), mlflow.get_registry_uri())



#working
@app.get("/")
async def read_root():
    return {"Tracking URI": mlflow.get_tracking_uri(),
            "Registry URI": mlflow.get_registry_uri()}

#working
@app.get("/models")
async def get_models_api():
    """Gets a list with model names"""
    model_list = mlflowclient.list_registered_models()
    model_list = [model.name for model in model_list]
    return model_list


#working
@app.post("/trainConv")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Creates a model based on hyperparameters and trains it."""
    model_name = data.model_name
    background_tasks.add_task(
        Trainer.train_model_task_conv, model_name)
    return {"result": "Training task started"}


#working
@app.post("/trainGCNGeometric")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Creates a model based on hyperparameters and trains it."""
    model_name = data.model_name
    background_tasks.add_task(
        Trainer.train_model_task_geometric, model_name)
    return {"result": "Training task started"}


@app.post("/trainGCNDGL")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Creates a model based on hyperparameters and trains it."""
    model_name = data.model_name
    background_tasks.add_task(
        Trainer.train_model_task_dgl, model_name)
    return {"result": "Training task started"}


#working if the model has been logged before
'''@param run_id: run id of the registered model
          model_name: name of the registererd model
'''
@app.post("/register")
async def register_api(data: RegisterApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    run_id = data.run_id
    model_name = data.model_name
    client = MlflowClient()
    result = client.create_model_version(
        name=model_name,
        source="s3://mlflow/1/{run_id}",
        run_id= run_id,
    )
    #mlflow.register_model(
            #"s3://mlflow/1/{run_id}",
            #model_name)
            
    return {"result": "Model Registered"}


#not working
@app.post("/registerByName")
async def register_api(data: RegisterByNameApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    model_name = data.model_name  
    mlflow.pytorch.log_model(
        pytorch_model=model_name,
        artifact_path="model",
        registered_model_name="test2",
    )
    return {"result": "Model Registered"}



@app.post("/predict_by_runid_gemoetric")
async def predictById_api(data: PredictIdApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    """Predicts on the provided image"""
    run_id = data.run_id
    # Load image from file
    data1, dataset = load_cora()
    # Resize the image to your desired size
    # Postprocess result
    logged_model = "runs:/920759fe16aa4a98bad0417aa0511564/model" #.format(run_id)
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pytorch.load_model(logged_model)   
    loaded_model.eval()
    out = loaded_model.predict(data1.x, data1.edge_index)
    pred = out.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred[data1.test_mask] == data1.y[data1.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data1.test_mask.sum())  # Derive ratio of correct predictions.
    return {"test_acc":test_acc}


@app.post("/predict_by_runid_dgl")
async def predictById_api(data: PredictIdApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    """Predicts on the provided image"""
    run_id = data.run_id
    # Load image from file
    g, dataset = load_cora_dgl()
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    test_mask = g.ndata["test_mask"]    # Resize the image to your desired size
    # Postprocess result
    logged_model = "runs:/6e0b696f4cc74e4991d4136b3970e17c/model" #.format(run_id)
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pytorch.load_model(logged_model)   
    logits = loaded_model(g, features)
    # Compute prediction
    pred = logits.argmax(1)
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    print(test_acc)
    return {"test_acc": float(test_acc)}



#working only with pyfunc, why?
'''@param run_id: run_id of the registered model
          img: path to image
'''
@app.post("/predict_by_runid_conv")
async def predictById_api(data: PredictIdApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    """Predicts on the provided image"""
    run_id = data.run_id
    # Load image from file
    image = Image.open(img)
    # Resize the image to your desired size
    img = np.array(image, dtype=np.float32).flatten()[np.newaxis, ...]/255
    # Postprocess result
    logged_model = "runs:/df2990c840c344b4bf8b7f9464fe1380/model" #.format(run_id)
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pytorch.load_model(logged_model)   
    #model = mlflow.pyfunc.load_model(
        #model_uri=f"models:/{model_name}/Production"
    #)
    # Preprocess the image
    with torch.no_grad():
        loaded_model.eval()
        pred = loaded_model.predict(img)
    
    print(pred)
    res = int(np.argmax(pred[0]))
    return {"result": res}




@app.post("/predict_serve_conv")
async def predictById_api(data: PredictIdApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    """Predicts on the provided image"""
    host = '0.0.0.0'
    port = '8001' 
    url = f'http://{host}:{port}/invocations' 
    headers = {'Content-Type': 'application/json',} 
    # test_data is a Pandas dataframe with data for testing the ML model
    run_id = data.run_id
    # Load image from file
    image = Image.open(img)
    # Resize the image to your desired size
    img = np.array(image, dtype=np.float32).flatten()[np.newaxis, ...]/255
    http_data = img.to_json(orient='split') 
    r = requests.post(url=url, headers=headers, data=http_data) 
    print(f'Predictions: {r.text}')
    res = int(np.argmax(r.text))
    return {"result": res}


#not working
'''@param model_name: name of the registered model
          img: path to image
'''
@app.post("/predict_by_name")
async def predictByName_api(data: PredictNameApiData):
    """Predicts on the provided image"""
    img = data.input_image
    model_name = data.model_name
    mlflow.set_tracking_uri("http://mlflow:5000")
    # Fetch the last model in production
    # Load image from file
    image = Image.open(img)
    # Resize the image to your desired size
    img = np.array(image, dtype=np.float32).flatten()[np.newaxis, ...]/255
    # Postprocess result
    model_uri = f"models:/{model_name}/1"
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("--")
    #model = mlflow.pyfunc.load_model(
        #model_uri=f"models:/{model_name}/Production"
    #)
    pred = loaded_model.predict(img)
    print(pred)
    res = int(np.argmax(pred[0]))
    return {"result": res}



#working
'''@param model_name: name of the registered model
          version: version of the model, it could be empty, then all versions are selected
'''
@app.post("/delete")
async def delete_model_api(data: DeleteApiData):
    model_name = data.model_name
    version = data.model_version
    if version is None:
        # Delete all versions
        mlflowclient.delete_registered_model(name=model_name)
        response = {"result": f"Deleted all versions of model {model_name}"}
    elif isinstance(version, list):
        for v in version:
            mlflowclient.delete_model_version(name=model_name, version=v)
        response = {
            "result": f"Deleted versions {version} of model {model_name}"}
    else:
        mlflowclient.delete_model_version(name=model_name, version=version)
        response = {
            "result": f"Deleted version {version} of model {model_name}"}
    return response



#working
'''@param model_name: name of the registered model
          ver: version of the model, it could be empty
          stage_to: next stage
'''
@app.post("/transition")
async def transition_model_api(data: TransitionApiData):
    model_name = data.model_name
    ver = data.version
    stage_to = data.transition_stage
    client = MlflowClient()
    client.transition_model_version_stage(
    name=model_name, version=ver, stage= stage_to
    )
    return  {"result": "Model Updated"}


#not working, i have to call function inside train 
@app.post("/saveModel")
async def transition_model_api(data: SaveModelApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    model_name = data.model_name
    mlflow.pytorch.save_model(model_name, "model")
    return  {"result": "Model Saved"}



#working (save an existing model stored in a pcikle file)
'''@param filename: path where the model is stored
          model_name: name to give when registered
'''
@app.post("/registerSavedModel")
async def transition_model_api(data: RegisterSavedModelApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    filename = data.filename
    model_name = data.model_name
    loaded_model = pickle.load(open(filename, "rb"))
    # log and register the model using MLflow scikit-learn API
    print("--")
    mlflow.pytorch.log_model(
        loaded_model,
        "pytorch",
        registered_model_name=model_name,
    )
    return  {"result": "Model Saved"}


#working
'''@param old_name: old name of a registered model
          model_name: new name of registered model
'''
@app.post("/renameMmodel")
async def transition_model_api(data: RenameModelApiData):
    mlflow.set_tracking_uri("http://mlflow:5000")
    old_name = data.old_name
    new_name = data.new_name
    client = MlflowClient()
    client.rename_registered_model(
    name= old_name,
    new_name=new_name,
    )
    return  {"result": "Model   Renamed"}



#runner = bentoml.mlflow.get("conv/1:latest").to_runner()

#loaded_model = mlflow.pytorch.load_model("models:/conv1/1")
#bentoml.pytorch.save_model("iris_clf", loaded_model)
#runner = bentoml.mlflow.import_model(
    #'mlflow_pytorch_mnist',
   # "models:/conv1/1",
    #signatures={'predict': {'batchable': True}}
    #)

        

        # make predictions with BentoML runner

#model_runner_2.init_local()
#bento_model = bentoml.mlflow.import_model('mlflow_pytorch_mnist', "models:/conv1/1")
#mnist_runner = bentoml.mlflow.get('bento_model:latest').to_runner()
#loaded_model = mlflow.sklearn.load_model("models:/conv1/1")
#bentoml.sklearn.save_model("conv1", loaded_model)
#runner = bentoml.mlflow.get('conv1:latest').to_runner()

#runner = bentoml.mlflow.get('mlflow_pytorch_mnist').to_runner()


#runner = bentoml.mlflow.import_model(
            #"mlflow_pytorch_mnist",
            ##"models:/conv1/1",
            #signatures={"predict": {"batchable": True}},
        #).to_runner()
#runner = bentoml.mlflow.get("mlflow_pytorch_mnist:latest").to_runner()
bento_model_2 = bentoml.mlflow.import_model(
    "mlflow_pytorch",
    "models:/conv1/1",
     signatures={"predict": {"batchable": True}},
 )
print("Model imported to BentoML: %s" % bento_model_2)
model_runner_2 = bentoml.mlflow.get("mlflow_pytorch:latest").to_runner()

        # make predictions with BentoML runner
svc = bentoml.Service("conv1", runners=[model_runner_2])
svc.mount_asgi_app(FastAPI)

@app.post("/predict")
async def predict_image():
    image = Image.open("api/img_4.jpg")
    img_arr = np.array(image)/255.0
    input_arr = np.expand_dims(img_arr, 0).astype("float32")
    output_tensor = model_runner_2.predict.run(input_arr)
    res = int(np.argmax(output_tensor.text))
    return {"result": res}
    #return output_tensor.numpy()
    #bento_model = bentoml.mlflow.import_model("mlflow_pytorch_mnist", "models:/conv1/1")
    
    #svc = bentoml.Service(
        #name="pytorch_mnist_demo",
        #runners=[runner]
    #)
    # Resize the image to your desired size
    #img = np.array(image, dtype=np.float32).flatten()[np.newaxis, ...]/255
    #output_tensor = model_runner_2.predict.async_run(img)
    #res = int(np.argmax(output_tensor[0]))
    #return output_tensor.numpy()
    #return model_runner_2.predict.async_run(img)

def ciao():
    ciao = "ciao"
    return "ciao"