# Bound-Conditioned Latent Inference for Progressive Image Compression (BLI)

Official evaluation code for **["Bound-Conditioned Latent Inference for Progressive Image Compression"](https://arxiv.org/abs/2610.30966) (NeurIPS 2026)**.

## 1. Requirements

- Ubuntu 20.04 recommended
- Python 3.11
- PyTorch 2.6.0 + CUDA 12.6
- CompressAI 1.2.8
- Other packages in `environment.yml`

Create and activate the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate bli
```

The environment file includes the CUDA 12.6 PyTorch wheel index. Other configurations have not been tested.

## 2. Pretrained checkpoints

Download the four checkpoints using the links below, and place them in the `checkpoint/` directory.

| Checkpoint | Trained objective | Download |
| --- | --- | --- |
| `BLI_psnr.pth` | PSNR | [Google Drive](https://drive.google.com/file/d/12rmyZbbKnoyll0k4u9a13fKliMCPBmvi/view?usp=sharing) |
| `BLI_ssim.pth` | MS-SSIM | [Google Drive](https://drive.google.com/file/d/1y_zyKhfzTZBNReIe0ybqOdeM7v4LjznF/view?usp=sharing) |
| `FTIC_psnr.pth` | PSNR | [Google Drive](https://drive.google.com/file/d/1d928G5lXz9ZYQuYbrUt5UxpPXvjsbkXo/view?usp=sharing) |
| `FTIC_ssim.pth` | MS-SSIM | [Google Drive](https://drive.google.com/file/d/1RpRCtOfiJ9w6j9xc65Vg4GW5Ir7EQMoB/view?usp=sharing) |

Expected directory layout:

```text
checkpoint/
├── BLI_psnr.pth
├── BLI_ssim.pth
├── FTIC_psnr.pth
└── FTIC_ssim.pth
```

The PSNR evaluation uses `BLI_psnr.pth` and `FTIC_psnr.pth`; the MS-SSIM evaluation uses `BLI_ssim.pth` and `FTIC_ssim.pth`.

## 3. Evaluation

Run from the repository root so that relative checkpoint paths resolve correctly.

### Default evaluation (PSNR-trained models)

Only `--data-dir` is required when using the default settings:

```bash
python eval.py --data-dir /path/to/test_images
```

Defaults: `--model psnr`, `--result-dir result`, `--L-list 4 4 6 16 24`.

### MS-SSIM-trained models

```bash
python eval.py --model ssim --data-dir /path/to/test_images
```

### Custom refinement steps

```bash
python eval.py --data-dir /path/to/test_images --L-list 4 4 6 16 24
```

`--L-list` specifies the number of refinement steps for trit-planes 5, 4, 3, 2, and 1, in that order.

## 4. Reference evaluation metrics

The `metrics/` directory contains reference CSV files for BLI, DPICT, and CTC, all implemented on the same [FTIC](https://proceedings.iclr.cc/paper_files/paper/2024/hash/81a0a519bf7ef7757ff46df2065e7051-Abstract-Conference.html) backbone (PSNR- and MS-SSIM-trained variants).

## 5. Citation

Please cite the following paper when you use this repository. Thanks!

```bibtex
@inproceedings{2026_NeurIPS_jang,
    author    = {Jang, Jaeseok and Jeon, Seungmin and Choi, Kwang Pyo and Kim, Chang-Su},
    title     = {Bound-Conditioned Latent Inference for Progressive Image Compression},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2026}
}
```

## 6. License

See [MIT License](LICENSE)
