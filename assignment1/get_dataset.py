import urllib.request
import tarfile
import os

url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
fileobj = urllib.request.urlopen(url)

dataset_path = os.path.join(os.path.dirname(__file__), 'dataset')

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)

print('Dataset Downloading')
with tarfile.open(fileobj=fileobj, mode="r|gz") as tar:
    tar.extractall(path=dataset_path)
