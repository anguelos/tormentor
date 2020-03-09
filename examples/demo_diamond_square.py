import timeit

import fargv
from matplotlib import pyplot as plt

from diamond_square import diamond_square

params = {
    "device": "cpu",
    "roughness": .5,
    "do_show": False,
    "replicates": 1,
    "recursion_depth": 10,
    "repeat": 100
}

if __name__ == "__main__":
    params, _ = fargv.fargv(params, return_named_tuple=True)
    duration = timeit.timeit(setup="from diamond_square import diamond_square",
                             stmt="img = diamond_square(recursion_depth={}, device={})".format(params.recursion_depth,
                                                                                               repr(params.device)),
                             number=params.repeat)
    print(f"Created {(2**10+1)**2/1000000} MPixels {params.repeat} times in {duration} sec.")
    if params.do_show:
        img = diamond_square(recursion_depth=params.recursion_depth, device="cpu")
        plt.imshow(img[0, 0, :, :])
        plt.colorbar()
        plt.show()
