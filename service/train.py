import numpy as np
import random
import pickle
import os

from keras.src.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.src.optimizers import Adam
import matplotlib.pyplot as plt
import cv2

from model.cnn import SimpleCNNNet
from utils import utils_paths


class TrainService:

    @classmethod
    def _save_model(cls, args, model, lb):
        model.save(args["model"])
        f = open(args["labels"], "wb")
        f.write(pickle.dumps(lb))
        f.close()

    @classmethod
    def _save_plot_png(cls, args, kpi, history):
        n = np.arange(0, kpi["epochs"])
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(n, history.history["loss"], label="loss")
        plt.plot(n, history.history["val_loss"], label="val_loss")
        plt.plot(n, history.history["accuracy"], label="accuracy")
        plt.plot(n, history.history["val_accuracy"], label="val_accuracy")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(args["plot"])

    def train(self, args: dict, kpi: dict):
        print("开始读取数据")
        # load data
        data, labels = self._load_data(kpi, args["dataset"])
        # data segmentation
        (train_x, test_x, train_y, test_y) = train_test_split(data, labels, test_size=kpi["test_size"], random_state=42)

        lb = LabelBinarizer()
        train_y = lb.fit_transform(train_y)
        test_y = lb.transform(test_y)

        # establishing convolutional neural networks
        model = SimpleCNNNet.build(classes=len(lb.classes_), kpi=kpi)

        # optimizer
        opt = Adam(learning_rate=kpi["init_lr"], decay=kpi["init_lr"] / kpi["epochs"])
        model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        train_x = train_x.reshape((-1, kpi["width"], kpi["height"], 3))
        test_x = test_x.reshape((-1, kpi["width"], kpi["height"], 3))

        # data enhancement
        if kpi["open_data_enhancement"]:
            aug = ImageDataGenerator(
                rotation_range=30, width_shift_range=0.1,
                height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                horizontal_flip=True, fill_mode="nearest"
            )
            history = model.fit_generator(
                aug.flow(train_x, train_y, batch_size=kpi["batch_size"]), validation_data=(test_x, test_y),
                steps_per_epoch=len(train_x) // kpi["batch_size"],
                epochs=kpi["epochs"], verbose=1
            )
        else:
            history = model.fit(
                train_x, train_y, validation_data=(test_x, test_y), epochs=kpi["epochs"], batch_size=kpi["batch_size"]
            )

        # test network model
        predictions = model.predict(test_x, batch_size=kpi["batch_size"])
        print(classification_report(test_y.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

        # save result curve
        self._save_plot_png(args, kpi, history)

        # save model
        self._save_model(args, model, lb)

    @classmethod
    def _load_data(cls, kpi, dataset):
        data = []
        labels = []
        image_paths = sorted(list(utils_paths.list_images(dataset)))
        random.seed(42)
        random.shuffle(image_paths)

        for image_path in image_paths:
            org_image = cv2.imread(image_path)
            if org_image is None:
                # 抛弃异常图片
                print(f"img imread error: {image_path}")
                os.remove(image_path)
                continue

            image = cv2.resize(org_image, (kpi["width"], kpi["height"])).flatten()
            data.append(image)

            label = image_path.split(os.path.sep)[-2]
            labels.append(label)

        data = np.array(data, dtype="float") / 255.0
        labels = np.array(labels)

        return data, labels
