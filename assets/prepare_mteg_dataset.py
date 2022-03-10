import os
import glob
from PIL import Image

help_msg = """
Example usage:
python prepare_mteg_dataset.py --gt_dir ./gtFine/ --img_dir ./imgFine --output_dir ./mteg_dataset
"""


def load_resized_img(path):
    return Image.open(path).convert('RGB').resize((384, 256))


def check_matching_pair(segmap_path, photo_path):
    segmap_identifier = os.path.basename(segmap_path).replace('_gtFine_color', '')
    photo_identifier = os.path.basename(photo_path).replace('_leftImg8bit', '')

    assert segmap_identifier == photo_identifier, \
        "[%s] and [%s] don't seem to be matching. Aborting." % (segmap_path, photo_path)


def process_mteg(gt_dir, img_dir, output_dir, phase):
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    # os.makedirs(savedir, exist_ok=True)
    os.makedirs(savedir + 'A', exist_ok=True)
    os.makedirs(savedir + 'B', exist_ok=True)
    print("Directory structure prepared at %s" % output_dir)

    # segmap_expr = os.path.join(gt_dir, phase) + "*_gtFine_color.png"
    segmap_expr = os.path.join(gt_dir, phase, "*", "*_gtFine_color.png")
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

# photo_expr = os.path.join(img_dir, phase) + "*_leftImg8bit.png"
photo_expr = os.path.join(img_dir, phase, "*", "*_leftImg8bit.png")
photo_paths = glob.glob(photo_expr)
photo_paths = sorted(photo_paths)

print(len(segmap_paths), len(photo_paths))
assert len(segmap_paths) == len(photo_paths), \
    "%d images that match [%s], and %d images that match [%s]. Aborting." % (
    len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)

for i, (segmap_path, photo_path) in enumerate(zip(segmap_paths, photo_paths)):
    check_matching_pair(segmap_path, photo_path)
    segmap = load_resized_img(segmap_path)
    photo = load_resized_img(photo_path)

# data for pix2pix where the two images are placed side-by-side
# sidebyside = Image.new('RGB', (512, 256))
# sidebyside.paste(segmap, (256, 0))
# sidebyside.paste(photo, (0, 0))
# savepath = os.path.join(savedir, "%d.jpg" % i)
# sidebyside.save(savepath, format='JPEG', subsampling=0, quality=100)

# data for cyclegan where the two images are stored at two distinct directories
savepath = os.path.join(savedir + 'A', "%d.png" % i)
photo.save(savepath, format='PNG', subsampling=0, quality=100)
savepath = os.path.join(savedir + 'B', "%d.png" % i)
segmap.save(savepath, format='PNG', subsampling=0, quality=100)

if i % (len(segmap_paths) // 10) == 0:
    print("%d / %d: last image saved at %s, " % (i, len(segmap_paths), savepath))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type=str, required=False, default='./gtFine',
                        help='Path to the MTEG gtFine directory.')
    parser.add_argument('--img_dir', type=str, required=False, default='./imgFine',
                        help='Path to the MTEG imgFine directory.')
    parser.add_argument('--output_dir', type=str, required=False,
                        default='./mteg_dataset',
                        help='Directory the output images will be written to.')
    opt = parser.parse_args()

print(help_msg)

print('Preparing MTEG Dataset for val phase')
process_mteg(opt.gt_dir, opt.img_dir, opt.output_dir, "val")
print('Preparing MTEG Dataset for train phase')
process_mteg(opt.gt_dir, opt.img_dir, opt.output_dir, "train")

print('Done')