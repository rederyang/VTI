# download COCO dataset to data dir
wget http://images.cocodataset.org/zips/train2014.zip -P /workspace/data/COCO2014
wget http://images.cocodataset.org/zips/val2014.zip -P /workspace/data/COCO2014

# prepare env
conda create -yn vti python=3.9
conda activate vti
git clone url/to/repo.git
cd repo
pip install -r requirement.txt
# other dependencies
pip install git+https://github.com/haotian-liu/LLaVA.git
pip install datasets==2.14.6
pip install pyarrow==14.0.2
pip install protobuf
pip install openai
