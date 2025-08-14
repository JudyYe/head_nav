import os
import os.path as osp
from glob import glob

import imageio
from fire import Fire
from projectaria_tools.core.sensor_data import TimeDomain
from tqdm import tqdm

from preprocess.utils import AriaData


def batch_save_images(raw_dir):
    seq_list = glob(osp.join(raw_dir, "*.vrs"))
    save_dir = raw_dir.replace("/raw", "/images")
    for seq in tqdm(seq_list):
        seq = osp.basename(seq).split(".")[0]

        save_file = osp.join(save_dir, f"{seq}.mp4")
        save_images(seq, save_file)


def save_images(seq="1", save_file="", raw_dir=""):
    dataset = AriaData(seq, downsample=4, data_dir=raw_dir)

    fps = 10
    index_list = dataset.data_provider.get_timestamps_ns(
        dataset.rgb_stream_id, TimeDomain.DEVICE_TIME
    )

    raw_image_list = []
    image_list = []
    for i in range(len(index_list)):
        img = dataset.get_image_by_time(index_list[i])
        raw_img = dataset.get_raw_image_by_time(index_list[i])

        image_list.append(img)
        raw_image_list.append(raw_img)

    os.makedirs(osp.dirname(save_file), exist_ok=True)
    imageio.mimwrite(save_file, image_list, fps=fps)
    print(f"saved {len(image_list)} images to {save_file}")


if __name__ == "__main__":
    Fire(batch_save_images)