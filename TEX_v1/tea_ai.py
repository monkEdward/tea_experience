from model import cnn_model as cm
from preprocess import img_processor as imp

tea_related = []

def guess(img, model, dataset):
    predicted_class = cm.predict_class(img, model, dataset)

    return predicted_class in tea_related


def main_creation_model():
    #TODO dataset loading and preprocessing

    dataset, val_dataset = imp.dataset_loader()
    model, history = cm.model_creation(dataset, val_dataset)
    cm.save_model(model)

    return dataset


if __name__ == '__main__':
    dataset = main_creation_model()
    model = cm.load_model()

    #TODO prediction of other pictures
    img = ''
    if guess(img, model, dataset):
        print('Tea related image')
    else:
        print('Not related image')