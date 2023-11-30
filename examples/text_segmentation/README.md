Experiments presented in the [tormentor paper](https://arxiv.org/pdf/2204.03776.pdf) can be reproduced here.

* Install tormentor if it is not installed
```bash
pip3 install tormentor=0.1.3 --user
```

* Add extra requirements.
```bash
pip3 install $(cat requirements.txt)
```

* Download DIBCO mirror if some of the dibco URLs are down
```bash
python3 -m pip install git+https://github.com/moeb/rpack
pip3 install --user gdown
FILEID=19K7bw1x8CVhn--ks4BRvQPqJe8zPDI93
FILENAME=all_dibco_data.tar.bz2
#gdown https://drive.google.com/uc?id="${FILEID}"
(mkdir -p tmp;cd tmp;gdown https://drive.google.com/uc?id="${FILEID}";tar -xpvjf "../${FILENAME}")
```

* Create Synthetic sudo pages
```bash
wget http://www.sls.hawaii.edu/bley-vroman/brown.txt -O ./data/brown.txt

./bin/synth.py
```

* Train Segmentation IUNET
```bash
PYTHONPATH="./src" ./bin/train -mode train -resume_fname ./tmp/dibco_iunet.pt 
```
* Evaluate Segmentation IUNET (Attention trainset included in tests)
```bash
./bin/test_all.sh  ./tmp/dibco_iunet.pt
```



