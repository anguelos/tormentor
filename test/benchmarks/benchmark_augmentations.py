import torch
import tormentor
import torchvision
import timeit
import pickle

batch_sizes = [1, 10]
image_sizes = [32, 224]
point_counts = [1, 10, 100]
channel_counts = [1, 3]
devices = ["cpu", "cuda:0", "cuda:1"]

augmentations_cls_list = []

for cls in [tormentor.SpatialImageAugmentation, tormentor.StaticImageAugmentation]:
    augmentations_cls_list += cls.__subclasses__()

#augmentations_cls_list=[tormentor.Rotate]

image_measurements = {}
pc_measurements = {}
image_and_pc_measurements = {}

for augmentation_cls in augmentations_cls_list:
    for batch_size in batch_sizes:
        for img_size in image_sizes:
            for point_count in point_counts:
                for channel_count in channel_counts:
                    for device in devices:
                        img = torch.rand(batch_size, channel_count, img_size, img_size, device=device)
                        pc = [(torch.rand(point_count, device=device) * img_size, torch.rand(point_count, device=device) * img_size,) for _ in range(batch_size)]
                        augmentation = augmentation_cls()
                        setup_str=""
                        run_pc = "augmentation(pc, img, compute_img=False)"
                        run_pc_and_img = "augmentation(pc, img, compute_img=True)"
                        run_img = "augmentation(img)"

                        #pc_duration = timeit.timeit(stmt=run_pc, setup=setup_str, number=5)
                        params = (augmentation_cls, batch_size, img_size, point_count, channel_count, device)
                        image_measurements[params] = timeit.timeit(stmt=run_img, setup=setup_str, number=5, globals = locals())
                        pc_measurements[params] = timeit.timeit(stmt=run_pc, setup=setup_str, number=5, globals = locals())
                        image_and_pc_measurements[params] = timeit.timeit(stmt=run_pc_and_img, setup=setup_str, number=5, globals = locals())
                        print(params, " -> ", image_measurements[params], pc_measurements[params], image_and_pc_measurements[params] )

        #pickle.dump({"img": image_measurements, "pc": pc_measurements, "img_pc":image_measurements},open("/tmp/bench.pickle","wb"))