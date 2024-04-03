from isegm.utils.exp_imports.default import *
from isegm.data.datasets.drone_avalanche import DroneAvalancheDataset
MODEL_NAME = 'avalanche_hrnet18'

def main(cfg):
    model, model_cfg = init_model(cfg)


def init_model(weight_path):
    #model_cfg = edict()
    #model_cfg.crop_size = (600, 600)
    #model_cfg.num_max_points = 24

    model = HRNetModel(width=18, ocr_width=64, with_aux_output=True, use_leaky_relu=True,
                       use_rgb_conv=False, use_disks=True, norm_radius=5, with_prev_mask=True)

    model.to(cfg.device)
    model.apply(initializer.XavierGluon(rnd_type='gaussian', magnitude=2.0))
    #model.feature_extractor.load_pretrained_weights(cfg.IMAGENET_PRETRAINED_MODELS.HRNETV2_W18)
    model.feature_extractor.load_pretrained_weights(weight_path.weights)














if __name__ == '__main__':
    main()
    print("success")