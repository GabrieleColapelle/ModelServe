import mlflow
import torchvision.transforms as transforms
from ml.models import Net, GCN, GCNdgl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
import pickle
import numpy as np
from ml.data import load_cora, load_cora_dgl
import torch.nn.functional as F
import bentoml

class Trainer():
    def train_model_task_conv(model_name: str):
        """Tasks that trains the model. This is supposed to be running in the background
        Since it's a heavy computation it's better to use a stronger task runner like Celery
        For the simplicity I kept it as a fastapi background task"""
        mlflow.set_tracking_uri("http://mlflow:5000")
        # Set MLflow tracking
        mlflow.set_experiment("MNIST")
        mlflow.end_run()
        with mlflow.start_run():
            print(mlflow.get_artifact_uri)
            #Log hyperparameters
            #mlflow.log_params(hyperparams)
            transform = transforms.Compose([
                                    transforms.ToTensor()
            ])
            # Loading Data and splitting it into train and validation data
            train = datasets.MNIST('', train = True, transform = transform, download = True)
            train, valid = random_split(train,[50000,10000])
            
            # Create Dataloader of the above tensor with batch size = 32
            trainloader = DataLoader(train, batch_size=32)
            validloader = DataLoader(valid, batch_size=32)
            print("Training model")
            run_id = mlflow.active_run().info.run_id
            modello = Net()
            # Declaring Criterion and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(modello.parameters(), lr = 0.01)
            
            # Training with Validation
            epochs = 5
            min_valid_loss = np.inf
            
            for e in range(epochs):
                train_loss = 0.0
                for data, labels in trainloader:
                    # Transfer Data to GPU if available
                    if torch.cuda.is_available():
                        data, labels = data.cuda(), labels.cuda()
                    
                    # Clear the gradients
                    optimizer.zero_grad()
                    # Forward Pass
                    target = modello(data)
                    # Find the Loss
                    loss = criterion(target,labels)
                    # Calculate gradients
                    loss.backward()
                    # Update Weights
                    optimizer.step()
                    # Calculate Loss
                    train_loss += loss.item()
                
                valid_loss = 0.0
                modello.eval()     # Optional when not using Model Specific layer
                for data, labels in validloader:
                    # Transfer Data to GPU if available
                    if torch.cuda.is_available():
                        data, labels = data.cuda(), labels.cuda()
                
                    # Forward Pass
                    target = modello(data)
                    # Find the Loss
                    loss = criterion(target,labels)
                    # Calculate Loss
                    valid_loss += loss.item()
                
                if min_valid_loss > valid_loss:
                    print(f'Validation Loss Decreased')
                    print('New validation loss ' + str(valid_loss))
                    min_valid_loss = valid_loss
                    print('best_loss'  + str(min_valid_loss))

            run_id = (mlflow.active_run().info.run_id)
            #log of the model
            #mlflow.pytorch.log_model(modello, "model")
            mlflow.pytorch.log_model(
                pytorch_model= modello,
                artifact_path="model",
                registered_model_name="conv1",
                )
            
            #bento_model = bentoml.mlflow.import_model("conv1", "models:/conv1/1")
            #save the model in docker filesystem as pkl file
            #filename = "conv_model.pkl"
            #pickle.dump(modello, open(filename, "wb"))
            #log and save at the same time 
            #mlflow.register_model(
                #"s3://mlflow/1/{run_id}",
                #"testmodel01")

            mlflow.end_run()   



    def train_model_task_geometric(model_name: str):
        """Tasks that trains the model. This is supposed to be running in the background
        Since it's a heavy computation it's better to use a stronger task runner like Celery
        For the simplicity I kept it as a fastapi background task"""
        mlflow.set_tracking_uri("http://mlflow:5000")
        # Set MLflow tracking
        mlflow.set_experiment("CORA")
        mlflow.end_run()
        with mlflow.start_run():
            print(mlflow.get_artifact_uri)
            data, dataset = load_cora()
            print("Training model")
            run_id = mlflow.active_run().info.run_id
            num_feat = dataset.num_features
            num_classes = dataset.num_classes
            model = GCN(hidden_channels=16, num_feat = num_feat, num_classes = num_classes)
            epochs = 5
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss()
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()  # Clear gradients.
                out = model(data.x, data.edge_index)  # Perform a single forward pass.
                loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
                loss.backward()  # Derive gradients.
                optimizer.step()  # Update parameters based on gradients.
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            
            run_id = (mlflow.active_run().info.run_id)
            #log of the model
            mlflow.pytorch.log_model(model, "model")
            mlflow.end_run()


    def train_model_task_dgl(model_name: str):
        """Tasks that trains the model. This is supposed to be running in the background
        Since it's a heavy computation it's better to use a stronger task runner like Celery
        For the simplicity I kept it as a fastapi background task"""
        mlflow.set_tracking_uri("http://mlflow:5000")
        # Set MLflow tracking
        mlflow.set_experiment("CORA")
        mlflow.end_run()
        with mlflow.start_run():
            print(mlflow.get_artifact_uri)
            g, dataset = load_cora_dgl()
            best_val_acc = 0
            best_test_acc = 0
            features = g.ndata["feat"]
            labels = g.ndata["label"]
            train_mask = g.ndata["train_mask"]
            val_mask = g.ndata["val_mask"]
            test_mask = g.ndata["test_mask"]
            print("Training model")
            model = GCNdgl(g.ndata["feat"].shape[1], 16, dataset.num_classes)
            epochs = 5
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            for epoch in range(epochs):
                # Forward
                logits = model(g, features)
                # Compute prediction
                pred = logits.argmax(1)
                # Compute loss
                # Note that you should only compute the losses of the nodes in the training set.
                loss = F.cross_entropy(logits[train_mask], labels[train_mask])
                # Compute accuracy on training/validation/test
                train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
                val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
                # Save the best validation accuracy and the corresponding test accuracy.
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            run_id = (mlflow.active_run().info.run_id)
            #log of the model
            mlflow.pytorch.log_model(model, "model")
            mlflow.end_run()