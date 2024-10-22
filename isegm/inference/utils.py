from datetime import timedelta
from pathlib import Path

import torch
import numpy as np

from isegm.data.datasets.drone_avalanche import DroneAvalancheDataset
from isegm.data.datasets import GrabCutDataset, BerkeleyDataset, DavisDataset, SBDEvaluationDataset, PascalVocDataset, AvalancheDataset #, DroneAvalancheDataset
from isegm.utils.serialization import load_model


def get_time_metrics(all_ious, elapsed_time):
    n_images = len(all_ious)
    n_clicks = sum(map(len, all_ious))

    mean_spc = elapsed_time / n_clicks
    mean_spi = elapsed_time / n_images

    return mean_spc, mean_spi


def load_is_model(checkpoint, device, **kwargs):
    if isinstance(checkpoint, (str, Path)): #thats the
        state_dict = torch.load(checkpoint, map_location='cpu')
    else:
        state_dict = checkpoint

    if isinstance(state_dict, list):
        model = load_single_is_model(state_dict[0], device, **kwargs)
        models = [load_single_is_model(x, device, **kwargs) for x in state_dict]

        return model, models
    else: #thats the path
        return load_single_is_model(state_dict, device, **kwargs)


def load_single_is_model(state_dict, device, **kwargs):
    model = load_model(state_dict['config'], **kwargs)
    model.load_state_dict(state_dict['state_dict'], strict=False)

    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    model.eval()

    return model


def get_dataset(dataset_name, cfg):
    if dataset_name == 'GrabCut':
        dataset = GrabCutDataset(cfg.GRABCUT_PATH)
    elif dataset_name == 'Berkeley':
        dataset = BerkeleyDataset(cfg.BERKELEY_PATH)
    elif dataset_name == 'DAVIS':
        dataset = DavisDataset(cfg.DAVIS_PATH)
    elif dataset_name == 'SBD':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH)
    elif dataset_name == 'SBD_Train':
        dataset = SBDEvaluationDataset(cfg.SBD_PATH, split='train')
    elif dataset_name == 'PascalVOC':
        dataset = PascalVocDataset(cfg.PASCALVOC_PATH, split='test')
    elif dataset_name == 'COCO_MVal':
        dataset = DavisDataset(cfg.COCO_MVAL_PATH)
    elif dataset_name == 'Avalanche1':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_1)
    elif dataset_name == 'Avalanche2':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_2)
    elif dataset_name == 'Avalanche3':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_3)
    elif dataset_name == 'Avalanche3a':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_4)
    elif dataset_name == 'Avalanche3b':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_5)
    elif dataset_name == 'Avalanche4':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_6)
    elif dataset_name == 'Avalanche_train':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_7)
    elif dataset_name == 'Avalanche_vali':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_8)
    elif dataset_name == 'Avalanche_DG':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_9)
    elif dataset_name == 'Train_ours_uibk':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_10)
    elif dataset_name == 'Vali_ours_uibk':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_11)
    elif dataset_name == 'Avalanche_uibk_test':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_12)
    elif dataset_name == 'Avalanche_uibk_test1': #test data without glide snow cracks
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_13)
    elif dataset_name == 'Avalanche_uibk_vali_combo':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_14)
    elif dataset_name == 'Avalanche_uibk_train':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_15)
    elif dataset_name == 'Avalanche_uibk_vali1':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_16)
    elif dataset_name == 'User_study':
        dataset = AvalancheDataset(cfg.AVALANCHE_PATH_17)
    elif dataset_name == 'Small_dataset':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_18)
    elif dataset_name == 'DS_v2_1m':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_19)
    elif dataset_name == 'DS_v2_0p5m':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_20)
    elif dataset_name == 'DS_v2_0p5m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_21)
    elif dataset_name == 'DS_v2_0p1m':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_22)
    elif dataset_name == 'DS_v2_0p1m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_23)
    elif dataset_name == 'DS_v2_1m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_24)
    elif dataset_name == 'DS_v3_0p2m_train':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_25)
    elif dataset_name == 'DS_v3_0p2m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_26)
    elif dataset_name == 'DS_v3_0p5m_train':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_27)
    elif dataset_name == 'DS_v3_0p5m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_28)
    elif dataset_name == 'DS_v3_1m_train':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_29)
    elif dataset_name == 'DS_v3_1m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_30)
    elif dataset_name == 'DS_v3_2m_train':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_31)
    elif dataset_name == 'DS_v3_2m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_32)
    elif dataset_name == 'DS_v3_5m_train':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_33)
    elif dataset_name == 'DS_v3_5m_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_34)
    elif dataset_name == 'DS_v3_0p5m_NoBlur_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_36)
    elif dataset_name == 'DS_v3_0p5m_NoBlur_2_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_37)
    elif dataset_name == 'DS_v3_0p5m_NoBlur_3_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_38)
    elif dataset_name == 'DS_v3_0p5m_DSM_only_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_39)
    elif dataset_name == 'DS_v3_0p5m_DSM_only_max4k_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_40)
    elif dataset_name == 'ds_v3_0p5m_DSM_only_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_41)
    elif dataset_name == 'ds_v3_0p5m_DSM_hillshade_norm_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_42)
    elif dataset_name == 'ds_v3_0p5m_withDSM_hillshade_RGH_channel_norm_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_43)
    elif dataset_name == 'ds_v3_0p5m_withDSM_hillshade_RHB_channel_norm_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_44)
    elif dataset_name == 'ds_v3_0p5m_withDSM_hillshade_HGB_channel_norm_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_45)
    elif dataset_name == 'ds_v3_0p5m_Aug_test_set':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_46)
    elif dataset_name == 'ds_v3_0p5m_gB_sig0p5_channel_norm_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_47)
    elif dataset_name == 'ds_v3_0p5m_gB_sig0p5_RGH_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_48)
    elif dataset_name == 'ds_v3_0p2m_gB_sig0p5_RGH_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_49)
    elif dataset_name == 'ds_v3_1m_gB_sig0p5_RGH_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_50)
    elif dataset_name == 'ds_v3_1m_NoBlur_4_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_51)
    elif dataset_name == 'ds_v3_0p5m_NoBlur_4_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_52)
    elif dataset_name == 'ds_v3_0p2m_NoBlur_4_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_53)
    elif dataset_name == 'ds_v3_0p2m_NoBlur_RGH_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_54)
    elif dataset_name == 'ds_v3_0p5m_NoBlur_RGH_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_55)
    elif dataset_name == 'ds_v3_1m_NoBlur_RGH_test':
        dataset = DroneAvalancheDataset(cfg.AVALANCHE_PATH_56)

    else:
        dataset = None

    return dataset


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union


def compute_noc_metric(all_ious, iou_thrs, max_clicks=20):
    def _get_noc(iou_arr, iou_thr):
        vals = iou_arr >= iou_thr
        return np.argmax(vals) + 1 if np.any(vals) else max_clicks

    noc_list = []
    over_max_list = []
    for iou_thr in iou_thrs:
        scores_arr = np.array([_get_noc(iou_arr, iou_thr)
                               for iou_arr in all_ious], dtype=np.int)

        score = scores_arr.mean()
        over_max = (scores_arr == max_clicks).sum()

        noc_list.append(score)
        over_max_list.append(over_max)

    return noc_list, over_max_list


def find_checkpoint(weights_folder, checkpoint_name):
    weights_folder = Path(weights_folder)
    if ':' in checkpoint_name:
        model_name, checkpoint_name = checkpoint_name.split(':')
        models_candidates = [x for x in weights_folder.glob(f'{model_name}*') if x.is_dir()]
        assert len(models_candidates) == 1
        model_folder = models_candidates[0]
    else:
        model_folder = weights_folder

    if checkpoint_name.endswith('.pth'):
        if Path(checkpoint_name).exists():
            checkpoint_path = checkpoint_name
        else:
            checkpoint_path = weights_folder / checkpoint_name
    else:
        model_checkpoints = list(model_folder.rglob(f'{checkpoint_name}*.pth'))
        assert len(model_checkpoints) == 1
        checkpoint_path = model_checkpoints[0]

    return str(checkpoint_path)


def get_results_table(noc_list, over_max_list, brs_type, dataset_name, mean_spc, elapsed_time,
                      n_clicks=20, resize=None, model_name=None):
    table_header = (f'|{"BRS Type":^13}|{"Dataset":^11}|{"Image Size":^13}|'
                    f'{"NoC@80%":^9}|{"NoC@85%":^9}|{"NoC@90%":^9}|'
                    f'{">="+str(n_clicks)+"@85%":^9}|{">="+str(n_clicks)+"@90%":^9}|'
                    f'{"SPC,s":^7}|{"Time":^9}|')
    row_width = len(table_header)

    header = f'Eval results for model: {model_name}\n' if model_name is not None else ''
    header += '-' * row_width + '\n'
    header += table_header + '\n' + '-' * row_width

    if resize == None:
        resize = 'original'
    eval_time = str(timedelta(seconds=int(elapsed_time)))
    table_row = f'|{brs_type:^13}|{dataset_name:^11}|{str(resize):^13}|'
    table_row += f'{noc_list[0]:^9.2f}|'
    table_row += f'{noc_list[1]:^9.2f}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{noc_list[2]:^9.2f}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{over_max_list[1]:^9}|' if len(noc_list) > 1 else f'{"?":^9}|'
    table_row += f'{over_max_list[2]:^9}|' if len(noc_list) > 2 else f'{"?":^9}|'
    table_row += f'{mean_spc:^7.3f}|{eval_time:^9}|'

    return header, table_row