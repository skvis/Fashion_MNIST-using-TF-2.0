import os
import gzip
import shutil
import pandas as pd


def download_dataset():
    link_list = ['http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz']

    if not os.path.isdir('../input'):
        os.makedirs('../input')
        for item in link_list:
            os.system(f"wget --no-check-certificate {item} -O ../input/{item.split('/')[-1]}")
    else:
        for item in link_list:
            if not os.path.exists(f"../input/{item.split('/')[-1]}"):
                os.system(f"wget --no-check-certificate {item} -O ../input/{item.split('/')[-1]}")
            else:
                print('File already exits')


def extract_dataset():
    for filename in (os.listdir('../input/')):
        with gzip.open(f"../input/{filename}", 'r') as f_in:
            with open(f"../input/{filename.split('.')[0]}", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(f'../input/{filename}')


def convert(imgf, labelf, outf, n):
    f = open(imgf, "rb")
    o = open(outf, "w")
    l = open(labelf, "rb")

    f.read(16)
    l.read(8)
    images = []

    for _ in range(n):
        image = [ord(l.read(1))]
        for j in range(28*28):
            image.append(ord(f.read(1)))
        images.append(image)

    for image in images:
        o.write(",".join(str(pix) for pix in image)+"\n")
    f.close()
    o.close()
    l.close()


def append_columns_name(train_df, test_df):
    tr_df = pd.read_csv(train_df, names=['pixel'+str(i) for i in range(0, 785)])
    te_df = pd.read_csv(test_df, names=['pixel'+str(i) for i in range(0, 785)])
    tr_df.rename(columns={'pixel0': 'label'}, inplace=True)
    te_df.rename(columns={'pixel0': 'label'}, inplace=True)
    tr_df.to_csv(train_df, index=False)
    te_df.to_csv(test_df, index=False)


if __name__ == '__main__':
    download_dataset()

    extract_dataset()

    convert("../input/train-images-idx3-ubyte", "../input/train-labels-idx1-ubyte", "../input/fashion_mnist_train.csv", 60000)
    convert("../input/t10k-images-idx3-ubyte", "../input/t10k-labels-idx1-ubyte", "../input/fashion_mnist_test.csv", 10000)

    append_columns_name('../input/fashion_mnist_train.csv', '../input/fashion_mnist_test.csv')
