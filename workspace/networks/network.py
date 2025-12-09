from monai.networks.nets import SwinUNETR, UNet, AttentionUnet, DynUNet

def network(model_name, args):
    print(f'model: {model_name}')
    if model_name == 'unet3d':
        return UNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            channels=(64, 128, 256, 256),
            strides=(2, 2, 2),
            num_res_units=0,
            act='RELU',
            norm='BATCH'
        ).to(args.device)

    elif model_name == 'attention_unet':
        return AttentionUnet(
          spatial_dims=3,
          in_channels=args.in_channels,
          out_channels=args.out_channels,
          channels=(32, 64, 128, 256),
          strides=(2, 2, 2),
        ).to(args.device)

    elif model_name == 'swinunetr':
        depths = tuple(args.depths) if args.depths is not None else (2, 2, 2, 2)
        default_heads = (3, 6, 12, 24)
        if len(depths) <= len(default_heads):
            num_heads = default_heads[:len(depths)]
        else:
            num_heads = default_heads + tuple([default_heads[-1]] * (len(depths) - len(default_heads)))
        return SwinUNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            patch_size=args.patch_size,
            depths=depths,
            num_heads=num_heads,
            feature_size=args.feature_size,
            drop_rate=args.drop_rate,
            attn_drop_rate=getattr(args, "attn_drop_rate", 0.0),
            dropout_path_rate=getattr(args, "dropout_path_rate", 0.0),
            use_checkpoint=True,
            spatial_dims=3,
        ).to(args.device)
    
    elif model_name == 'DynUNet':
        return DynUNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=[[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]],
            strides=[[1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2]],
            upsample_kernel_size=[[2,2,2], [2,2,2], [2,2,2], [2,2,2]],
            filters=[16, 32, 64, 128, 256]
        ).to(args.device)
    
    else:
        raise ValueError(f'not found model name: {model_name}')

