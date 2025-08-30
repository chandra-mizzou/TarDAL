import argparse
import logging
from pathlib import Path
from typing import List, Tuple
import sys

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
from kornia.color import ycbcr_to_rgb
from kornia.geometry import resize as k_resize
import yaml

# allow running from any working directory by adding repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import from_dict
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device
from loader.utils.reader import gray_read, ycbcr_read, img_write


IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(folder: Path) -> dict:
    files = {}
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files[p.stem] = p
    return files


def intersect_pairs(ir_dir: Path, vi_dir: Path) -> List[Tuple[str, Path, Path]]:
    ir_map = list_images(ir_dir)
    vi_map = list_images(vi_dir)
    common = sorted(set(ir_map.keys()) & set(vi_map.keys()))
    if len(common) == 0:
        raise FileNotFoundError(f"No matching filenames between {ir_dir} and {vi_dir}.")
    return [(n, ir_map[n], vi_map[n]) for n in common]


def find_max_size(pairs: List[Tuple[str, Path, Path]]) -> torch.Size:
    max_h, max_w = -1, -1
    for _, ir_p, _ in pairs:
        img = gray_read(ir_p)
        max_h = max(max_h, img.shape[1])
        max_w = max(max_w, img.shape[2])
    return torch.Size((max_h, max_w))


class PairFolderDataset(Dataset):
    type = 'fuse'

    def __init__(self, pairs: List[Tuple[str, Path, Path]], visible_color: bool, max_size: torch.Size):
        super().__init__()
        self.pairs = pairs
        self.visible_color = visible_color
        self.max_size = max_size
        self.transform_fn = Resize(size=max_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict:
        name, ir_p, vi_p = self.pairs[index]
        ir = gray_read(ir_p)
        if self.visible_color:
            vi_y, cbcr = ycbcr_read(vi_p)
            s = ir.shape[1:]
            t = torch.cat([ir, vi_y, cbcr], dim=0)
            ir, vi, cbcr = torch.split(self.transform_fn(t), [1, 1, 2], dim=0)
            return {'name': name, 'ir': ir, 'vi': vi, 'cbcr': cbcr, 'shape': s}
        else:
            vi = gray_read(vi_p)
            s = ir.shape[1:]
            t = torch.cat([ir, vi], dim=0)
            ir, vi = torch.chunk(self.transform_fn(t), chunks=2, dim=0)
            return {'name': name, 'ir': ir, 'vi': vi, 'shape': s}

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        keys = batch[0].keys()
        collated = {}
        for k in keys:
            values = [b[k] for b in batch]
            collated[k] = values if isinstance(values[0], str) or isinstance(values[0], torch.Size) else torch.stack(values)
        return collated


def save_predictions(fus: torch.Tensor, names: List[str], shapes: List[torch.Size], out_dir: Path, visible_color: bool, cbcr: torch.Tensor | None, grayscale: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    if visible_color and not grayscale:
        fus = torch.cat([fus, cbcr], dim=1)
        fus = ycbcr_to_rgb(fus)
    for img_t, name, img_s in zip(fus, names, shapes):
        img_t = k_resize(img_t, img_s)
        out_p = out_dir / f"{name}.png"
        img_write(img_t, out_p)


def main():
    parser = argparse.ArgumentParser(description='TarDAL folder-based fusion of paired IR/VI images')
    parser.add_argument('--cfg', default='config/official/infer/tardal-dt.yaml', help='config file path')
    parser.add_argument('--ir_dir', required=True, help='directory of infrared images')
    parser.add_argument('--vi_dir', required=True, help='directory of visible images')
    parser.add_argument('--out_dir', default='runs/fused-folders', help='directory to save fused outputs')
    parser.add_argument('--visible_color', action='store_true', help='treat visible images as color (YCbCr)')
    parser.add_argument('--batch_size', type=int, default=None, help='override config.inference.batch_size')
    parser.add_argument('--num_workers', type=int, default=None, help='override config.inference.num_workers')
    parser.add_argument('--grayscale', action='store_true', help='force grayscale output regardless of visible mode')
    args = parser.parse_args()

    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level='INFO', format=log_f)
    logging.info('TarDAL folder-pair fusion start')

    config = yaml.safe_load(Path(args.cfg).open('r'))
    config = from_dict(config)

    if args.batch_size is not None:
        config.inference.batch_size = args.batch_size
    if args.num_workers is not None:
        config.inference.num_workers = args.num_workers
    if args.grayscale:
        config.inference.grayscale = True

    ir_dir = Path(args.ir_dir)
    vi_dir = Path(args.vi_dir)
    out_dir = Path(args.out_dir)

    pairs = intersect_pairs(ir_dir, vi_dir)
    max_size = find_max_size(pairs)
    dataset = PairFolderDataset(pairs=pairs, visible_color=args.visible_color, max_size=max_size)
    loader = DataLoader(
        dataset,
        batch_size=config.inference.batch_size,
        shuffle=False,
        collate_fn=PairFolderDataset.collate_fn,
        pin_memory=True,
        num_workers=config.inference.num_workers,
    )

    fuse = Fuse(config, mode='inference')

    with torch.inference_mode():
        for sample in loader:
            sample = dict_to_device(sample, fuse.device)
            fus = fuse.inference(ir=sample['ir'], vi=sample['vi'])
            cbcr = sample.get('cbcr', None)
            save_predictions(
                fus=fus,
                names=sample['name'],
                shapes=sample['shape'],
                out_dir=out_dir,
                visible_color=args.visible_color,
                cbcr=cbcr,
                grayscale=config.inference.grayscale,
            )


if __name__ == '__main__':
    main()

