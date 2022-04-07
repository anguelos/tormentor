Download DIBCO mirror if some of the dibco URLs are down
```bash
python3 -m pip install git+https://github.com/moeb/rpack
pip3 install --user gdown
FILEID=19K7bw1x8CVhn--ks4BRvQPqJe8zPDI93
FILENAME=all_dibco_data.tar.bz2
#gdown https://drive.google.com/uc?id="${FILEID}"
(mkdir -p tmp;cd tmp;gdown https://drive.google.com/uc?id="${FILEID}";tar -xpvjf "../${FILENAME}")
```
