import opendatasets as od
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmetrics import Accuracy

import ray
from ray import train
from ray.air import session
from ray.air.config import ScalingConfig
from ray.train.torch import TorchCheckpoint, TorchTrainer
from ray.data.preprocessors import Concatenator, Chain, StandardScaler
from ray import tune
from ray.tune.tuner import Tuner, TuneConfig
from ray.air.config import RunConfig
from ray.train.batch_predictor import BatchPredictor
from ray.train.torch import TorchPredictor
from ray.data.preprocessor import Preprocessor
from ray.data.aggregate import Max
from ray.data.datasource.partitioning import Partitioning
import cv2


# od.download(
#     "https://www.kaggle.com/datasets/wwymak/architecture-dataset")

HEIGHT = 320
WIDTH = 320

root = "architecture-dataset/arcDataset/"
partitioning = Partitioning("dir", field_names=["class"], base_dir=root)
dataset = ray.data.read_images(root, partitioning=partitioning, size=(HEIGHT, WIDTH), mode="RGB")
train_dataset, validation_dataset = dataset.train_test_split(test_size=0.05)

CLASS_ID_MAPPING = {'Achaemenid architecture': 0, 'American Foursquare architecture': 1, 'American craftsman style': 2, 'Ancient Egyptian architecture': 3, 'Art Deco architecture': 4, 'Art Nouveau architecture': 5, 'Baroque architecture': 6, 'Bauhaus architecture': 7, 'Beaux-Arts architecture': 8, 'Byzantine architecture': 9, 'Chicago school architecture': 10, 'Colonial architecture': 11, 'Deconstructivism': 12, 'Edwardian architecture': 13, 'Georgian architecture': 14, 'Gothic architecture': 15, 'Greek Revival architecture': 16, 'International style': 17, 'Novelty architecture': 18, 'Palladian architecture': 19, 'Postmodern architecture': 20, 'Queen Anne architecture': 21, 'Romanesque architecture': 22, 'Russian Revival architecture': 23, 'Tudor Revival architecture': 24}

class OpenCVPreprocessor(Preprocessor):
    def _fit(self, dataset):
        self.stats_ = None
    
    def _transform_numpy(self, df):
        images = df["image"]
        result = []
        for image in images:
            image = cv2.Canny(image=image, threshold1=100, threshold2=200) 
            image = cv2.resize(image, (HEIGHT,WIDTH), interpolation=cv2.INTER_CUBIC)
            result.append([image])
        result = np.array(result)
        df["image"] = result
        
        classes = df["class"]
        result = []
        for class_name in classes:
            result.append(self._hot_encode(class_name))
        result = np.array(result)
        df["class"] = result
        return df

    def _hot_encode(self, name):
        result = CLASS_ID_MAPPING[name]
        #result[] = 1
        return result
    
preprocessor = OpenCVPreprocessor()
#transformed = preprocessor.fit_transform(ds)

NUMBER_OF_CLASSES = len(CLASS_ID_MAPPING)

class CNN(nn.Module):
    def __init__(self, number_of_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 8)
        self.conv3 = nn.Conv2d(16, 32, 8)
        self.fc1 = nn.Linear(800, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, number_of_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def train_loop_per_worker(config):
    batch_size = config["batch_size"]
    lr = config["lr"]
    epochs = config["num_epochs"]
    number_of_classes = config["number_of_classes"]

    train_data = session.get_dataset_shard("train")
    model = CNN(number_of_classes)
    model = train.torch.prepare_model(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for _cur_epoch in range(epochs):
        for batch in train_data.iter_torch_batches(batch_size=batch_size, dtypes=torch.float32):
            inputs, labels = batch["image"], batch["class"]
            optimizer.zero_grad()
            predictions = model(inputs)
            train_loss = loss_fn(predictions, labels.type(torch.LongTensor))
            train_loss.backward()
            optimizer.step()
        loss = train_loss.item()
        session.report({"loss": loss}, checkpoint=TorchCheckpoint.from_model(model))
    
        accuracy = Accuracy(task="multiclass", num_classes=NUMBER_OF_CLASSES)

        session.report({"accuracy": accuracy(predictions, labels).numpy()}, checkpoint=TorchCheckpoint.from_model(model))

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "batch_size": 128,
        "num_epochs": 20,
        "lr": 0.01,
        "number_of_classes": NUMBER_OF_CLASSES
    },
    scaling_config=ScalingConfig(
        num_workers=6, 
        use_gpu=True
    ),
    datasets={"train": train_dataset},
    preprocessor=preprocessor,
)

result = trainer.fit()
print(f"Last result: {result.metrics}")