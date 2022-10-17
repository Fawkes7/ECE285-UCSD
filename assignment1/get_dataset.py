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
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(tar, path=dataset_path)
