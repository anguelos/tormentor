import torch
from torch import nn
import iunets
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

def tensor2pil(tensor):
    if len(tensor.size()) == 4:
        assert tensor.size(0) == 1
        tensor = tensor[0,:,:,:]
    if tensor.size(0) in (1,2):
        tensor = tensor[0, :, :]
    else:
        assert len(tensor.size()) == 2 or tensor.size(0) in (3, 4)
    array = np.uint8(tensor.cpu().numpy().astype("float") * 255)
    return Image.fromarray(array)

class BUNet(nn.Module):
    @staticmethod
    def resume(fname, **kwargs):
        try:
            if "device" in kwargs.keys():
                device = kwargs["device"]
            else:
                device = ""
            state_dict=torch.load(fname, map_location="cpu")
            constructor_params = state_dict["constructor_params"]
            del state_dict["constructor_params"]
            validation_epochs = state_dict["validation_epochs"]
            del state_dict["validation_epochs"]
            train_epochs = state_dict["train_epochs"]
            del state_dict["train_epochs"]
            args_history = state_dict["args_history"]
            del state_dict["args_history"]
            net = BUNet(**constructor_params)
            net.load_state_dict(state_dict)
            net.validation_epochs = validation_epochs
            net.train_epochs = train_epochs
            net.args_history = args_history
            net = net.to(device=device)
            return net
        except FileNotFoundError:
            return BUNet(**kwargs)

    def __init__(self, input_channels=3, target_channels=2, channels=(64, 128, 256, 384), stack_size=2, device="cuda") -> None:
        super().__init__()
        self.input2iunet = nn.Conv2d(in_channels=input_channels, out_channels=channels[0], kernel_size=3, padding=1)
        self.iunet2output = nn.Conv2d(in_channels=channels[0], out_channels=target_channels, kernel_size=3, padding=1)
        self.iunet = iunets.iUNet(in_channels=channels[0], channels=channels[1:], architecture=(stack_size,)*(len(channels)-1), dim=2)
        self.train_epochs = []
        self.validation_epochs = {}
        self.constructor_params = {"input_channels": input_channels, "target_channels":target_channels, "channels": channels, "stack_size":stack_size, "device":device}
        self.args_history={}
        self.to(device=device)


    def forward(self, x):
        x = self.input2iunet(x)
        x = self.iunet(x)
        x = self.iunet2output(x)
        return x
    

    def binarize(self, input, to_pil=False, threshold=False):
        if isinstance(input, torch.Tensor):
            with torch.no_grad():
                if len(input.size()) == 3:
                    input = torch.unsqueeze(input, 0)
                assert len(input.size()) == 4
                input = input.to(next(self.parameters()).device)
                output = self.forward(input)
                output = (F.softmax(output, dim=1)[0, 1, :,:])
                
                if threshold:
                    output = (output > .5).float("float")
                
                if to_pil:
                    return tensor2pil(output)
                else:
                    return output
        elif isinstance(input, DataLoader):
            results = []
            for sample_data in input:
                sample_input_image = sample_data[0]
                results.append(tuple([tensor2pil(s) for s in (self.binarize(sample_input_image),)+ tuple(sample_data)]))
            return results
        else:
            raise ValueError("Expect dataloader or dataset")


    def save(self, fname, args=None):
        state_dict = self.state_dict()
        state_dict["constructor_params"] = self.constructor_params
        state_dict["validation_epochs"] = self.validation_epochs
        state_dict["train_epochs"] = self.train_epochs
        state_dict["args_history"] = self.args_history
        torch.save(state_dict, fname)
