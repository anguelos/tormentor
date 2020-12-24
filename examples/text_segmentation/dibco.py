import zipfile
import rarfile
import py7zr
import torchvision
from PIL import Image, ImageOps
from io import BytesIO
from subprocess import getoutput as shell_stdout
import os
import errno
import torch


def _get_dict(compressed_stream, filter_gt=False, filter_nongt=False):
    assert not (filter_gt and filter_nongt)
    isimage = lambda x:x.split(".")[-1].lower() in ["tiff", "bmp", "jpg", "tif", "jpeg", "png"] and not "skel" in x.lower()
    isgt = lambda x: isimage(x) and ("gt" in x or "GT" in x)
    if isinstance(compressed_stream, py7zr.SevenZipFile):
        compressed_stream.reset()
        res_dict = compressed_stream.readall()
        names = res_dict.keys()
        if filter_gt:
            names = [n for n in names if not isgt(n)]
        if filter_nongt:
            names = [n for n in names if isgt(n)]
        return {name: res_dict[name] for name in names}
    elif isinstance(compressed_stream, rarfile.RarFile) or isinstance(compressed_stream, zipfile.ZipFile):
        names = compressed_stream.namelist()
        names = [n for n in names if isimage(n)]
        if filter_gt:
            names = [n for n in names if not isgt(n)]
        if filter_nongt:
            names = [n for n in names if isgt(n)]
        return {name: BytesIO(compressed_stream.read(compressed_stream.getinfo(name))) for name in names}
    else:
        raise ValueError("Filedescriptor must be one of [rar, zip, 7z]")


def extract(archive,root=None):
    if archive.endswith(".tar.gz"):
        if root is None:
            cmd="tar -xpvzf {}".format(archive)
        else:
            cmd = 'mkdir -p {};tar -xpvzf {} -C{}'.format(root,archive,root)
        output = shell_stdout(cmd)
    else:
        raise NotImplementedError()


def check_os_dependencies():
    program_list=["wget"]
    return all([shell_stdout("which "+prog) for prog in program_list])

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def resumable_download(url,save_dir):
    mkdir_p(save_dir)
    download_cmd = 'wget --directory-prefix=%s -c %s' % (save_dir, url)
    print("Downloading {} ... ".format(url))
    shell_stdout(download_cmd)
    print("done")
    return os.path.join(save_dir,url.split("/")[-1])


dibco_transform_gray_input = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    lambda x: torch.cat([x, 1 - x])
])

dibco_transform_color_input = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    #torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

dibco_transform_gt = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(),
    torchvision.transforms.ToTensor(),
    lambda x: torch.cat([x, 1 - x])
])


class Dibco:
    """Provides one or more of the `DIBCO <https://vc.ee.duth.gr/dibco2019/>` datasets.

    Os dependencies: Other than python packages, unrar and arepack CLI tools must be installed.
    In Ubuntu they can be installed with: sudo apt install unrar atool p7zip-full
    In order to concatenate two DIBCO datasets just add them:
    .. source :: python

        trainset = dibco.Dibco.Dibco2009() + dibco.Dibco.Dibco2013()
        valset = dibco.Dibco.Dibco2017() + dibco.Dibco.Dibco209()

    Each item is a tuple of an RGB PIL image and an Binary PIL image. The images are transformed by ``input_transform``
    and ``gt_transform``.
    """
    urls = {
        "2009_HW": ["https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBC02009_Test_images-handwritten.rar",
                    "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_handwritten.rar"],

        "2009_P": ["https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009_Test_images-printed.rar",
                   "https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_printed.rar"],

        "2010": ["http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_test_images.rar",
                 "http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_GT.rar"],

        "2011_P": ["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-machine_printed.rar"],
        "2011_HW": ["http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-handwritten.rar"],

        "2012": ["http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/dataset/H-DIBCO2012-dataset.rar"],

        "2013": ["http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/dataset/DIBCO2013-dataset.rar"],

        "2014": ["http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/original_images.rar",
                 "http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/GT.rar"],
        "2016": ["https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip",
                 "https://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip"],
        "2017": ["https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_Dataset.7z",
                 "https://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_GT.7z"],
        "2018": ["http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018_Dataset.zip",
                 "http://vc.ee.duth.gr/h-dibco2018/benchmark/dibco2018-GT.zip"],
        "2019A":["https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_dataset_trackA.zip",
                 "https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_gt_trackA.zip"],
        "2019B":["https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_dataset_trackB.zip",
                 "https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_GT_trackB.zip"]
    }

    @staticmethod
    def load_single_stream(compressed_stream):
        input_name2bs = _get_dict(compressed_stream, filter_gt=True)
        gt_name2bs = _get_dict(compressed_stream, filter_nongt=True)
        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in gt_name2bs.items()}
        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in input_name2bs.items()}
        assert set(id2gt.keys()) == set(id2in.keys())
        #id2gt = {k: ImageOps.invert(v.convert("RGB")).convert('1') for k, v in id2gt.items()}
        id2in = {k: v.convert("RGB") for k, v in id2in.items()}
        return {k: (id2in[k], id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def load_double_stream(input_compressed_stream, gt_compressed_stream):
        input_name2bs = _get_dict(input_compressed_stream)
        gt_name2bs = _get_dict(gt_compressed_stream)
        id2in = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in input_name2bs.items()}
        id2gt = {n.split("/")[-1].split("_")[0].split(".")[0]: Image.open(fd).copy() for n, fd in gt_name2bs.items()}
        assert set(id2gt.keys()) == set(id2in.keys())
        #id2gt = {k: ImageOps.invert(v.convert("RGB")).convert('1') for k, v in id2gt.items()}
        id2in = {k: v.convert("RGB") for k, v in id2in.items()}
        return {k: (id2in[k], id2gt[k]) for k in id2gt.keys()}

    @staticmethod
    def Dibco2009(**kwargs):
        kwargs["partitions"] = ["2009_HW", "2009_P"]
        return Dibco(**kwargs)


    @staticmethod
    def Dibco2010(**kwargs):
        kwargs["partitions"] = ["2010"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2011(**kwargs):
        kwargs["partitions"] = ["2011_P","2011_HW"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2012(**kwargs):
        kwargs["partitions"] = ["2012"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2013(**kwargs):
        kwargs["partitions"] = ["2013"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2014(**kwargs):
        kwargs["partitions"] = ["2014"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2016(**kwargs):
        kwargs["partitions"] = ["2016"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2017(**kwargs):
        kwargs["partitions"] = ["2017"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2018(**kwargs):
        kwargs["partitions"] = ["2018"]
        return Dibco(**kwargs)

    @staticmethod
    def Dibco2019(**kwargs):
        kwargs["partitions"] = ["2019A","2019B"]
        return Dibco(**kwargs)

    def __init__(self, partitions=["2009_HW", "2009_P"], root="./tmp/dibco", input_transform=dibco_transform_color_input, gt_transform=dibco_transform_gt, add_mask=False):
        self.input_transform = input_transform
        self.gt_transform = gt_transform
        self.root = root
        self.add_mask = add_mask
        data = {}
        for partition in partitions:
            for url in Dibco.urls[partition]:
                archive_fname = root + "/" + url.split("/")[-1]
                if not os.path.isfile(archive_fname):
                    resumable_download(url, root)
                else:
                    print(archive_fname," found in cache.")
            if len(Dibco.urls[partition]) == 2:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][0].split("/")[-1])
                    gt_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][1].split("/")[-1])
                    samples = {partition + "/" + k: v for k, v in Dibco.load_double_stream(input_rar, gt_rar).items()}
                    data.update(samples)
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname = root + "/" + Dibco.urls[partition][0].split("/")[-1]
                    zip_gt_fname = root + "/" + Dibco.urls[partition][1].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        input_zip = py7zr.SevenZipFile(zip_input_fname)
                    else:
                        input_zip = zipfile.ZipFile(zip_input_fname)
                    if zip_gt_fname.endswith("7z"):
                        gt_zip = py7zr.SevenZipFile(zip_gt_fname)
                    else:
                        gt_zip = zipfile.ZipFile(zip_gt_fname)
                    samples = {partition + "/" + k: v for k, v in Dibco.load_double_stream(input_zip, gt_zip).items()}
                    data.update(samples)
                else:
                    raise ValueError("Unknown file type")
            else:
                if Dibco.urls[partition][0].endswith(".rar"):
                    input_rar = rarfile.RarFile(root + "/" + Dibco.urls[partition][0].split("/")[-1])
                    samples = {partition + "/" + k: v for k, v in Dibco.load_single_stream(input_rar).items()}
                    data.update(samples)
                elif Dibco.urls[partition][0].endswith(".zip") or Dibco.urls[partition][0].endswith(".7z"):
                    zip_input_fname = root + "/" + Dibco.urls[partition][0].split("/")[-1]
                    if zip_input_fname.endswith("7z"):
                        # zip_input_fname = zip_input_fname[:-2] + "zip"
                        input_zip = py7zr.SevenZipFile(zip_input_fname)
                    else:
                        input_zip = zipfile.ZipFile(zip_input_fname)
                    samples = {partition + "/" + k: v for k, v in Dibco.load_single_stream(input_zip).items()}
                    data.update(samples)
                else:
                    raise ValueError("Unknown file type")
        id_data = list(data.items())
        self.sample_ids = [sample[0] for sample in id_data]
        self.inputs = [sample[1][0] for sample in id_data]
        self.gt = [sample[1][1] for sample in id_data]

    def __getitem__(self, item):
        input_img = self.input_transform(self.inputs[item])
        gt = self.gt_transform(self.gt[item])
        if self.add_mask:
            return input_img, gt, torch.ones_like(input_img[:1, :, :])
        else:
            return input_img, gt

    def __len__(self):
        return len(self.sample_ids)

    def __add__(self, other):
        res = Dibco(partitions=[])
        res.root = self.root
        res.input_transform = self.input_transform
        res.gt_transform = self.gt_transform
        res.sample_ids = self.sample_ids + other.sample_ids
        res.inputs = self.inputs + other.inputs
        res.gt = self.gt + other.gt
        return res
