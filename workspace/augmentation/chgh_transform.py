try:
    from monai.transforms import AddChanneld
except ImportError:  
    from monai.transforms import EnsureChannelFirstd as AddChanneld
from monai.transforms import (
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandAxisFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandCoarseDropoutd,
    RandZoomd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord
) 
try:
    from monai.transforms import EnsureDivisibled
except ImportError:
    EnsureDivisibled = None

try:
    from monai.transforms import RandElasticd
except ImportError:
    RandElasticd = None
try:
    from monai.transforms import RandBiasFieldd
except ImportError:
    RandBiasFieldd = None
try:
    from monai.transforms import RandHistogramShiftd
except ImportError:
    RandHistogramShiftd = None

def get_train_transform(args):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=args.b_min,
            b_max=args.b_max,
            clip=True,
        ),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=args.num_samples,
            image_key="image",
            image_threshold=0,
            allow_smaller=True,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=args.rand_flipd_prob,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=args.rand_flipd_prob,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=args.rand_flipd_prob,
        ),
        RandRotate90d(
            keys=["image", "label"],
            prob=args.rand_rotate90d_prob,
            max_k=3,
        ),
        RandAxisFlipd(
            keys=["image", "label"],
            prob=0.2,
        ),
        RandZoomd(
            keys=["image", "label"],
            prob=0.2,
            min_zoom=0.9,
            max_zoom=1.1,
            mode=("trilinear", "nearest"),
            keep_size=True,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=0.10,
            prob=args.rand_shift_intensityd_prob,
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=0.1,
            prob=args.rand_scale_intensityd_prob,
        ),
    ]

    if getattr(args, "strong_aug", False):
        transforms.extend([
            RandAffined(
                keys=["image", "label"],
                prob=0.35,
                rotate_range=(0.35, 0.35, 0.35),
                translate_range=(5, 5, 5),
                scale_range=(0.1, 0.1, 0.1),
                padding_mode="border",
                mode=("bilinear", "nearest"),
            ),
        ])
        if RandElasticd is not None:
            transforms.append(
                RandElasticd(
                    keys=["image", "label"],
                    prob=0.2,
                    sigma_range=(6, 9),
                    magnitude_range=(100, 300),
                    spacing=(16, 16, 16),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                )
            )
        transforms.extend([
            RandGaussianNoised(
                keys=["image"],
                prob=0.2,
                mean=0.0,
                std=0.01,
            ),
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.2,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.2,
                gamma=(0.7, 1.5),
            ),
        ])
        if RandBiasFieldd is not None:
            transforms.append(
                RandBiasFieldd(
                    keys=["image"],
                    prob=0.2,
                    coeff_range=(0.0, 0.1),
                )
            )
        if RandHistogramShiftd is not None:
            transforms.append(
                RandHistogramShiftd(
                    keys=["image"],
                    prob=0.2,
                    num_control_points=5,
                )
            )
        transforms.append(
            RandCoarseDropoutd(
                keys=["image"],
                prob=0.2,
                holes=5,
                spatial_size=32,
                fill_value=args.a_min,
            ),
        )

    transforms.append(ToTensord(keys=["image", "label"]))
    return Compose(transforms)


def get_val_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            ToTensord(keys=["image", "label"])
        ]
    )


def get_inf_transform(keys, args):
    if len(keys) == 2:
        # image and label
        mode = ("bilinear", "nearest")
    elif len(keys) == 3:
        # image and mutiple label
        mode = ("bilinear", "nearest", "nearest")
    else:
        # image
        mode = ("bilinear")
        
    return Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=mode,
            ),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=args.b_min, 
                b_max=args.b_max,
                clip=True,
                allow_missing_keys=True
            ),
            *( [EnsureDivisibled(keys=keys, k=32, allow_missing_keys=True)] if EnsureDivisibled else [] ),
            AddChanneld(keys=keys),
            ToTensord(keys=keys)
        ]
    )


def get_label_transform(keys=["label"]):
    return Compose(
        LoadImaged(keys=keys)
    )
