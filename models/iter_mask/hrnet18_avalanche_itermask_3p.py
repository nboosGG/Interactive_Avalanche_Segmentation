from isegm.utils.exp_imports.default import *

from isegm.data.datasets.drone_avalanche import DroneAvalancheDataset
MODEL_NAME = 'avalanche_hrnet18'


def main(cfg):
    model, model_cfg = init_model(cfg)
    train(model, cfg, model_cfg)

def init_model(cfg):
    model_cfg = edict()
    model_cfg.crop_size = (600, 600)
    model_cfg.num_max_points = 24

    model = HRNetModel(width=18, ocr_width=64, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=True, use_DSM=False)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    #model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)
    model.feature_extractor.load_pretrained_weights(cfg.weights)

    return model, model_cfg


def train(model, cfg, model_cfg):
    cfg.batch_size = 4 if cfg.batch_size < 1 else cfg.batch_size
    cfg.val_batch_size = cfg.batch_size
    crop_size = model_cfg.crop_size

    loss_cfg = edict()
    loss_cfg.instance_loss = NormalizedFocalLossSigmoid(alpha=0.5, gamma=2)
    loss_cfg.instance_loss_weight = 1 # default 1
    loss_cfg.instance_aux_loss = SigmoidBinaryCrossEntropyLoss()
    loss_cfg.instance_aux_loss_weight = 0.4 # default 0.4

    train_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)), # keep always
        HorizontalFlip(),
        Rotate(limit=10), #default 10
        ShiftScaleRotate(shift_limit=0.005, scale_limit=0,   #shift default 0.03
                         rotate_limit=(-3, 3), border_mode=0, p=0.75),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0) , # keep always
        RandomCrop(*crop_size), #keep always
        RandomBrightnessContrast(brightness_limit=(-0.25, 0.25), contrast_limit=(-0.15, 0.4), p=0.75), # brightness: -0.25, 0.25
        RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.75)
    ], p=1.0)

    val_augmentator = Compose([
        UniformRandomResize(scale_range=(0.75, 1.25)),
        PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0),
        RandomCrop(*crop_size)
    ], p=1.0)

    points_sampler = MultiPointSampler(model_cfg.num_max_points, prob_gamma=0.80,
                                       merge_objects_prob=0.15,
                                       max_num_merged_objects=2)
    
    

    #for key in cfg:
    #    print("item: ", key, cfg[key])

    
    trainset = DroneAvalancheDataset(
        cfg.ds_v3_0p5m_DSM_only_max4k_train, #cfg.AVALANCHE_TRAIN, #AVALANCHE_TRAIN
        split='train',
        augmentator=train_augmentator,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
    )

    valset = DroneAvalancheDataset(
        cfg.ds_v3_0p5m_DSM_only_max4k_train, #cfg.AVALANCHE_VALI, #AVALANCHE_VALI
        split='val',
        augmentator=val_augmentator,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
    )
    """

    trainset = AvalancheDataset(
        cfg.AVALANCHE_TRAIN, #AVALANCHE_TRAIN
        split='train',
        augmentator=train_augmentator,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
    )

    valset = AvalancheDataset(
        cfg.AVALANCHE_VALI, #AVALANCHE_VALI
        split='val',
        augmentator=val_augmentator,
        keep_background_prob=0.01,
        points_sampler=points_sampler,
    )
    """
    
   #Cosine LR
    optimizer_params = {
        'lr': 5e-4, 'betas': (0.9, 0.999), 'eps': 1e-8 #lr default 5e-4
    }
    lr_scheduler = partial(torch.optim.lr_scheduler.CosineAnnealingLR,T_max=100, eta_min=0) #T_max: max number of iterations, eta_min: minimum learning rate

    # copy of original with Multistep LR
    #optimizer_params = {
    #   'lr': 5e-6, 'betas': (0.9, 0.999), 'eps': 1e-8 #lr default 5e-6 , betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999)) eps: term added to the denominator to improve numerical stability (default: 1e-8)
    #}
    #lr_scheduler = partial(torch.optim.lr_scheduler.MultiStepLR,
                           #milestones=[50, 75], gamma=0.1) #decays the Lr by gamma/90% every step_size epochs

    trainer = ISTrainer(model, cfg, model_cfg, loss_cfg,
                        trainset, valset,
                        optimizer='adam',
                        optimizer_params=optimizer_params,
                        lr_scheduler=lr_scheduler,
                        checkpoint_interval=[(0, 5), (100, 1)],
                        image_dump_interval=1, #20
                        metrics=[AdaptiveIoU()],
                        max_interactive_points=model_cfg.num_max_points,
                        max_num_next_clicks=10) #10
    
    trainer.run(num_epochs=100)