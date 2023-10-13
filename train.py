from service import train_service

# input args
args = {
    "dataset": "./dataset",
    "model": f"./output/category.model",
    "labels": f"./output/category.pickle",
    "plot": f"./output/category.png",
}
kpi = dict(
    dropout=0.5,  # Dropout rate for the neural network
    stddev=0.02,  # Standard deviation for Gaussian initialization of weights in the neural network
    l2=0.01,  # L2 regularization parameter
    init_lr=0.001,  # Initial learning rate for gradient descent
    epochs=200,  # Number of times the entire training
    batch_size=32,
    width=32,
    height=32,
    depth=3,
    test_size=0.1,  # Proportion of the training set to be used for testing during training
    open_data_enhancement=True,
)

if __name__ == '__main__':
    train_service.train(args, kpi)
