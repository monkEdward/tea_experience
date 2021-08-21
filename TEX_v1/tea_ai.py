from model import cnn_model as cm
from preprocess import img_processor as imp

tea_related = ['accessory', 'cup', 'leaf', 'mug', 'pitcher', 'tea_bag', 'tea_package', 'tea_pot', 'teaset', 'tray_set']

def guess(img, model, dataset):
    predicted_class = cm.predict_class(img, model, dataset)

    return predicted_class in tea_related


def main_creation_model():

    dataset, test_set = imp.dataset_loader()
    train_ds, val_ds = imp.create_train_test_val_dataset(dataset, 64, 200)

    model, history = cm.model_creation(train_ds, val_ds)
    cm.save_model(model)

    return dataset


if __name__ == '__main__':
    training = 'True'

    if training:
        dataset = main_creation_model()
    else:
        dataset, test_set = imp.dataset_loader()
        model = cm.load_model()
        #TODO prediction of other pictures
        img = ''
        if guess(img, model, dataset):
            print('Tea related image')
        else:
            print('Not related image')