from keras.models import load_model
import pickle
import cv2


class IdentifyService:

    @classmethod
    def identify(cls, image_path: str):
        args = {
            "model": f"output/category.model",
            "labels": f"output/category.pickle",
            "width": 32,
            "height": 32,
            "image_url": image_path,
        }

        # load images
        image = cv2.imread(args["image_url"])
        image = cv2.resize(image, (args["width"], args["height"]))
        image = image.astype("float") / 255.0
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        # load model and labels
        model = load_model(args["model"])
        lb = pickle.loads(open(args["labels"], "rb").read())
        labels = list(lb.classes_)

        # prediction
        prediction = model.predict(image)
        i = prediction.argmax(axis=1)[0]
        identify_label = labels[i]
        prediction = [f"{str(round(p*100, 2))}%" for p in list(prediction[0])]
        identify_rate = prediction[i]
        prediction = dict(zip(labels, prediction))

        result = dict(
            identify_label=identify_label,
            identify_rate=identify_rate,
            prediction=prediction,
        )

        return result
