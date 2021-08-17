from model import cnn_model as cm
from preprocess import img_processor as imp


def main():
    imp.dataset_loader()


if __name__ == '__main__':
    main()