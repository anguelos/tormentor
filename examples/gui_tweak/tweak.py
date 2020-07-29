#!/usr/bin/env python3

import tormentor # used by the command line argument

import PIL
import PyQt5
import PyQt5.QtWidgets as qwidgets
import fargv
import torchvision
import time
from PyQt5.QtCore import Qt as qt
from PyQt5.QtCore import pyqtSlot


class UniformDistributionTweaker(qwidgets.QWidget):
    def __init__(self, name, distribution):
        super().__init__()
        self.options_range = (distribution.min[0].cpu().item(), distribution.max[0].cpu().item())
        self.distribution = distribution
        self.name = name
        min = self.distribution.min.cpu().item()
        max = self.distribution.max.cpu().item()
        self.steps = 300
        self.range_extremes = (min - (max - min), max + (max - min))
        self.initUI()

    def int2float(self, n):
        res = self.range_extremes[0] + (n / self.steps) * (self.range_extremes[1] - self.range_extremes[0])
        res = min(res, self.range_extremes[1])
        res = max(res, self.range_extremes[0])
        return res

    def update_txt(self):
        cur_min = self.int2float(self.min_slider.value())
        cur_max = self.int2float(self.max_slider.value())
        txt = f"tormentor.Uniform(({cur_min:.3},{cur_max:.3}))"
        self.display_values.setText(txt)

    def update_extremes(self):
        self.extreme_min_label.setText(f"{self.range_extremes[0]:.3}")
        self.extreme_max_label.setText(f"{self.range_extremes[1]:.3}")

    def update_distribution(self):
        cur_min = self.int2float(self.min_slider.value())
        cur_max = self.int2float(self.max_slider.value())
        self.distribution.min[0] = cur_min
        self.distribution.max[0] = cur_max

    @pyqtSlot(int)
    def max_values_changed(self, value):
        if self.min_slider.value() > value:
            self.max_slider.setValue(self.min_slider.value())
        self.update_txt()

    @pyqtSlot(int)
    def min_values_changed(self, value):
        if self.max_slider.value() < value:
            self.min_slider.setValue(self.max_slider.value())
        self.update_txt()

    def initUI(self):
        self.min_slider = qwidgets.QSlider(qt.Horizontal)
        self.min_slider.setMaximum(self.steps)
        self.max_slider = qwidgets.QSlider(qt.Horizontal)
        self.max_slider.setMaximum(self.steps)
        self.min_slider.setTickPosition(qwidgets.QSlider.TicksAbove)
        self.max_slider.setTickPosition(qwidgets.QSlider.TicksBelow)
        self.min_slider.setValue(self.steps / 3)
        self.max_slider.setValue(2 * self.steps / 3)
        self.display_values = qwidgets.QLineEdit()
        self.display_values.setReadOnly(True)

        self.extreme_min_label = qwidgets.QLabel()
        self.extreme_max_label = qwidgets.QLabel()
        self.name_label = qwidgets.QLabel(self.name)

        slider_vbox = qwidgets.QVBoxLayout()
        slider_vbox.addWidget(self.min_slider)
        slider_vbox.addWidget(self.max_slider)
        slider_hbox = qwidgets.QHBoxLayout()
        slider_hbox.addWidget(self.extreme_min_label)
        slider_hbox.addLayout(slider_vbox)
        slider_hbox.addWidget(self.extreme_max_label)
        vbox = qwidgets.QVBoxLayout()
        vbox.addWidget(self.name_label)
        vbox.addLayout(slider_hbox)
        vbox.addWidget(self.display_values)

        self.setLayout(vbox)
        self.min_slider.valueChanged.connect(self.min_values_changed)
        self.max_slider.valueChanged.connect(self.max_values_changed)

        self.update_extremes()
        self.update_txt()

        self.show()



class AugmentationTweaker(qwidgets.QWidget):
    def toPIL(self, tensor):
        if len(tensor.size()) == 3:
            return torchvision.transforms.ToPILImage()(tensor)
        elif len(tensor.size()) == 4:
            img_tensor = torchvision.utils.make_grid(tensor, nrow=self.nrow)
            return torchvision.transforms.ToPILImage()(img_tensor)
        else:
            raise ValueError()

    def __init__(self, augmentation_str, torch_img, replicates=12, nrow=4, show_input=True):
        super().__init__()
        self.show_input = show_input
        self.replicates = replicates
        self.nrow = nrow
        self.augmentation_type = eval(augmentation_str)
        self.augmentation_str = augmentation_str
        self.dist_list = list(self.augmentation_type.get_distributions(copy=False).items())
        self.torch_img = torch_img
        self.initUI()

    @pyqtSlot()
    def update_augmentation(self):
        for tweaker in self.tweakers:
            tweaker.update_distribution()
        batch_tensor = self.torch_img.unsqueeze(dim=0).repeat(self.replicates, 1, 1, 1)
        t = time.time()
        batch_tensor = self.augmentation_type()(batch_tensor)
        dur = time.time() - t
        log = f"{self.torch_img.size(-1)} x {self.torch_img.size(-2)} x {self.replicates} {1000*dur:.3} msec."
        self.setWindowTitle(log)
        if self.show_input:
            batch_tensor[0, :, :, :] = self.torch_img
        batch_tensor = batch_tensor.to("cpu")
        im = self.toPIL(batch_tensor)
        im = im.convert("RGBA")
        data = im.tobytes("raw", "RGBA")
        qim = PyQt5.QtGui.QImage(data, im.size[0], im.size[1], PyQt5.QtGui.QImage.Format_RGBA8888)
        self.img_label.setPixmap(PyQt5.QtGui.QPixmap.fromImage(qim))
        self.augmentation_str_box.setText(repr(self.augmentation_type))

    def initUI(self):
        self.tweakers = []
        vbox = qwidgets.QVBoxLayout()
        self.augmentation_str_box = qwidgets.QLineEdit()
        vbox.addWidget(self.augmentation_str_box)
        self.img_label = qwidgets.QLabel()
        vbox.addWidget(self.img_label)



        scroll_vbox = qwidgets.QVBoxLayout()
        group_box = qwidgets.QGroupBox()
        group_box.setLayout(scroll_vbox)
        sa = qwidgets.QScrollArea()
        sa.setWidgetResizable(True)
        #sa.setFixedHeight(400)
        sa.setWidget(group_box)
        vbox.addWidget(sa)
        for name, distribution in self.dist_list:
            if isinstance(distribution, tormentor.Uniform):
                tweaker = UniformDistributionTweaker(name, distribution)
                scroll_vbox.addWidget(tweaker)
                self.tweakers.append(tweaker)
            else:
                scroll_vbox.addWidget(qwidgets.QLabel(f"{name}: {repr(distribution)}"))
        redraw_button = qwidgets.QPushButton("Redraw")
        redraw_button.clicked.connect(self.update_augmentation)

        vbox.addWidget(redraw_button)
        self.setLayout(vbox)
        self.update_augmentation()
        self.show()


params = {"image": "./Lenna.png",
          "replicates": 8,
          "ncol": 4,
          "device": "cuda",
          "show_input": True,
          "augmentation": "tormentor.Wrap"}

if __name__ == "__main__":
    params, _ = fargv.fargv(params, return_named_tuple=True)
    img = torchvision.transforms.ToTensor()(PIL.Image.open(params.image)).to(params.device)
    app = qwidgets.QApplication([])
    d = AugmentationTweaker(params.augmentation, img, params.replicates, params.ncol, params.show_input)
    d.show()
    app.exec_()
