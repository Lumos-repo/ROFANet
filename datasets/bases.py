from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import numpy as np
import os


def inject_noisy_correspondence(dataset, noisy_rate, noisy_file =None):
    logger = logging.getLogger("RDE.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions  = [i[4] for i in dataset_copy]
    imagesoc=[i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i]== i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i],image_ids[i],images[i],imagesoc[i],captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))

    return dataset, np.array(real_correspondeces)

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("RDE.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result
 

class ImageDataset(Dataset):
    def __init__(self, image_pids, img_papaths,img_ocpaths, transform=None):
        self.image_pids = image_pids
        self.img_papaths = img_papaths
        self.img_ocpaths=img_ocpaths
        self.transform = transform
        self.patch_size = 16
        self.N_expected = 192  # 你的 N=192
        self.parsing_dataset = 'lip' # lip/atr
        self.lut = build_merge_lut(self.parsing_dataset)
    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_papath,img_ocpath = self.image_pids[index], self.img_papaths[index],self.img_ocpaths[index]
        # imgpa = read_image(img_papath)
        imgoc=read_image(img_ocpath)
        parsing_index = load_parsing_index_png(img_papath)  # [H,W] long

        # 3) 先对 imgoc 做 transform，得到最终输入尺寸 (H_img,W_img)
        if self.transform is not None:
            imgoc = self.transform(imgoc)   # [3,H,W]

        # 4) 把 parsing_index 转成 patch labels (N,)
        parsing_labels_patch, gh, gw = pixel_mask_to_patch_labels(
            parsing_mask_index=parsing_index,
            img_tensor=imgoc,
            patch_size=self.patch_size,
            N_expected=self.N_expected,
            lut=self.lut
        )
        return pid,parsing_labels_patch,imgoc


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
  
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def build_merge_lut(dataset: str = "lip", device="cpu"):
    """
    ReID-friendly parsing merge LUT
    Returns:
        lut: LongTensor [256] mapping original parsing label -> {0(bg),1(head),2(torso_arms),3(legs)}
    """
    lut = torch.zeros(256, dtype=torch.long, device=device)

    if dataset.lower() == "lip":
        # LIP labels (20 classes)
        # 0 Background

        # ===== head =====
        head = [
            1,   # Hat
            2,   # Hair
            13,  # Face
            4    # Sunglasses (有的 LIP 版本里存在)
        ]

        # ===== torso + arms + upper accessories =====
        torso_arms = [
            5,   # Upper-clothes
            6,   # Dress
            7,   # Coat
            14,  # Left-arm
            15,  # Right-arm
            3,   # Glove
            11,  # Scarf
            12,  # Skirt (更偏上半身视觉)
            8    # Socks（部分实现里算上半身附件）
        ]

        # ===== legs + lower-body details =====
        legs = [
            9,   # Pants
            10,  # Jumpsuits
            16,  # Left-leg
            17,  # Right-leg
            18,  # Left-shoe
            19   # Right-shoe
        ]

    elif dataset.lower() == "atr":
        # ATR labels (18 classes)
        # 0 Background

        head = [
            1,   # Hat
            2,   # Hair
            11   # Face
        ]

        torso_arms = [
            4,   # Upper-clothes
            7,   # Dress
            14,  # Left-arm
            15,  # Right-arm
            3,   # Sunglasses
            10   # Bag
        ]

        legs = [
            6,   # Pants
            12,  # Left-leg
            13,  # Right-leg
            8,   # Left-shoe
            9    # Right-shoe
        ]
    else:
        raise ValueError(f"Unsupported parsing dataset: {dataset}")

    for k in head:
        lut[k] = 1
    for k in torso_arms:
        lut[k] = 2
    for k in legs:
        lut[k] = 3

    # truly background / occluder remains 0
    return lut



# ---------- 2) 像素 mask -> patch labels ----------
def pixel_mask_to_patch_labels(
    parsing_mask_index: torch.Tensor,
    img_tensor: torch.Tensor,
    patch_size: int = 16,
    N_expected: int = 192,
    lut: torch.Tensor = None,
):
    """
    parsing_mask_index: LongTensor [H, W]  (pixel-wise class indices, e.g., LIP 0..19)
    img_tensor: FloatTensor [3, H_img, W_img]  (your transformed imgoc tensor)
    patch_size: usually 16
    N_expected: your patch token count (192)
    lut: mapping original label -> {0,1,2,3} (LongTensor [256])

    Returns:
        parsing_labels_patch: LongTensor [N_expected] in {0,1,2,3}
        gh, gw: patch grid
    """
    assert img_tensor.ndim == 3, "img_tensor should be [C,H,W]"
    H_img, W_img = int(img_tensor.shape[1]), int(img_tensor.shape[2])

    # patch grid
    gh, gw = H_img // patch_size, W_img // patch_size
    if gh * gw != N_expected:
        raise ValueError(
            f"Patch grid mismatch: H={H_img},W={W_img},patch={patch_size} => "
            f"gh={gh},gw={gw}, gh*gw={gh*gw} but expected N={N_expected}. "
            f"Please verify img size / patch_size."
        )

    # resize parsing mask to the same resolution as img_tensor (if not already)
    if parsing_mask_index.shape[-2:] != (H_img, W_img):
        m = parsing_mask_index[None, None].float()  # [1,1,H,W]
        m = F.interpolate(m, size=(H_img, W_img), mode="nearest")
        parsing_mask_index = m[0, 0].long()

    # downsample to patch grid (nearest) -> [gh, gw]
    m = parsing_mask_index[None, None].float()
    m_small = F.interpolate(m, size=(gh, gw), mode="nearest")[0, 0].long()  # [gh,gw]

    # merge to 4 classes
    if lut is not None:
        m_small = lut[m_small.clamp(0, 255)]

    parsing_labels_patch = m_small.reshape(-1).long()  # [N]
    return parsing_labels_patch, gh, gw


# ---------- 3) 正确读取 palette-index png ----------
def load_parsing_index_png(path: str) -> torch.Tensor:
    """
    Load parsing png as index map (NOT RGB).
    Returns LongTensor [H,W]
    """
    img = Image.open(path)  # keep palette mode if exists
    # Do NOT convert to RGB!
    arr = np.array(img, dtype=np.int64)  # pixel values are indices
    return torch.from_numpy(arr).long()

class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.txt_aug = args.txt_aug
        self.img_aug = args.img_aug
       
        self.dataset, self.real_correspondences = inject_noisy_correspondence(dataset,args.noisy_rate,args.noisy_file)
        self.tokenizer = SimpleTokenizer()
        self.patch_size = getattr(args, "patch_size", 16)
        self.N_expected = getattr(args, "num_patches", 192)  # 你的 N=192
        self.parsing_dataset = getattr(args, "parsing_dataset", "lip")  # lip/atr
        self.lut = build_merge_lut(self.parsing_dataset)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_papath, img_ocpath, caption = self.dataset[index]

        # 1) 先读遮挡图（RGB正常读）
        imgoc = read_image(img_ocpath)

        # 2) 读 parsing mask：必须 index 方式
        parsing_index = load_parsing_index_png(img_papath)  # [H,W] long

        # 3) 先对 imgoc 做 transform，得到最终输入尺寸 (H_img,W_img)
        if self.transform is not None:
            imgoc = self.transform(imgoc)   # [3,H,W]

        # 4) 把 parsing_index 转成 patch labels (N,)
        parsing_labels_patch, gh, gw = pixel_mask_to_patch_labels(
            parsing_mask_index=parsing_index,
            img_tensor=imgoc,
            patch_size=self.patch_size,
            N_expected=self.N_expected,
            lut=self.lut
        )
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        if self.txt_aug:
            caption_tokens = self.txt_data_aug(caption_tokens.cpu().numpy())
        # print(parsing_labels_patch)
        # print("parsing_labels_patch shape:", parsing_labels_patch.shape)
        # print("unique labels:", torch.unique(parsing_labels_patch))
        ret = {
        'pids': pid,
        'image_ids': image_id,
        'parsing_labels_patch': parsing_labels_patch,
        'imagesoc':imgoc,
        'caption_ids': caption_tokens,
        'index':index,
        }

        return ret

    def txt_data_aug(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                # mask token with 15% probability
                if prob < 0.20:
                    prob /= 0.20
                    # 50% randomly change token to mask token
                    if prob < 0.6:
                        aug_tokens.append(mask) 
                    # 20% randomly change token to random token
                    elif prob < 0.8:
                        aug_tokens.append(random.choice(token_range)) # -> rest 10% randomly keep current token
                    else:
                        None # # 30% randomly remove
                else:
                    # no masking token (will be ignored by loss function later)
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens)