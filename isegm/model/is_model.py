import torch
import torch.nn as nn
import numpy as np

from isegm.model.ops import DistMaps, ScaleLayer, BatchImageNormalize
from isegm.model.modifiers import LRMult


class ISModel(nn.Module):
    def __init__(self, use_rgb_conv=True, with_aux_output=False,
                 norm_radius=260, use_disks=False, cpu_dist_maps=False,
                 clicks_groups=None, with_prev_mask=False, use_leaky_relu=False,
                 binary_prev_mask=False, conv_extend=False, norm_layer=nn.BatchNorm2d,
                 norm_mean_std=([.485, .456, .406], [.229, .224, .225]),
                 use_DSM=False):
        super().__init__()
        self.with_aux_output = with_aux_output
        self.clicks_groups = clicks_groups
        self.with_prev_mask = with_prev_mask
        self.binary_prev_mask = binary_prev_mask
        self.normalization = BatchImageNormalize(norm_mean_std[0], norm_mean_std[1])
        self.use_DSM = use_DSM

        print("use DSM:", self.use_DSM)

        self.coord_feature_ch = 2
        if clicks_groups is not None:
            self.coord_feature_ch *= len(clicks_groups)

        if self.with_prev_mask:
            self.coord_feature_ch += 1

        if use_rgb_conv:
            rgb_conv_layers = [
                nn.Conv2d(in_channels=3 + self.coord_feature_ch, out_channels=6 + self.coord_feature_ch, kernel_size=1),
                norm_layer(6 + self.coord_feature_ch),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=6 + self.coord_feature_ch, out_channels=3, kernel_size=1)
            ]
            self.rgb_conv = nn.Sequential(*rgb_conv_layers)
        elif conv_extend:
            self.rgb_conv = None
            self.maps_transform = nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=64,
                                            kernel_size=3, stride=2, padding=1)
            self.maps_transform.apply(LRMult(0.1))
        else:
            self.rgb_conv = None
            mt_layers = [
                nn.Conv2d(in_channels=self.coord_feature_ch, out_channels=16, kernel_size=1),
                nn.LeakyReLU(negative_slope=0.2) if use_leaky_relu else nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=1),
                ScaleLayer(init_value=0.05, lr_mult=1)
            ]
            self.maps_transform = nn.Sequential(*mt_layers)

        if self.clicks_groups is not None:
            self.dist_maps = nn.ModuleList()
            for click_radius in self.clicks_groups:
                self.dist_maps.append(DistMaps(norm_radius=click_radius, spatial_scale=1.0,
                                               cpu_mode=cpu_dist_maps, use_disks=use_disks))
        else:
            self.dist_maps = DistMaps(norm_radius=norm_radius, spatial_scale=1.0,
                                      cpu_mode=cpu_dist_maps, use_disks=use_disks)

    def forward(self, image, points):
        #print("in model forward, image shape: ", image.shape, "points shape:", points.shape)
        
        image, prev_mask, dsm = self.prepare_input(image)
        #print("in is module 69: ", image.shape, prev_mask.shape, dsm.shape)
        #print("in is module 69: ", image.shape, prev_mask.shape, torch.sum(dsm), torch.mean(dsm), torch.max(dsm))
        #image = torch.cat((image,dsm), dim=1)
        coord_features = self.get_coord_features(image, prev_mask, points)
        #print("shapes coord features: ", coord_features.shape, image.shape)

        if self.rgb_conv is not None: #not used
            x = self.rgb_conv(torch.cat((image, coord_features), dim=1))
            assert(False and "suddenly using rgb_conv, wtf??")
            outputs = self.backbone_forward(x)
        else: #used
            #print("not using rgb_conv, cord_features: ", coord_features.shape)
            
            coord_features = self.maps_transform(coord_features) #1x1 conv 2->64
            #print("input shapes for backbone_forward: ", image.shape, coord_features.shape)
            #print("coord featrues shape after transform: ", coord_features.shape)
            outputs = self.backbone_forward(image, dsm, coord_features)

        outputs['instances'] = nn.functional.interpolate(outputs['instances'], size=image.size()[2:],
                                                         mode='bilinear', align_corners=True)
        if self.with_aux_output:
            outputs['instances_aux'] = nn.functional.interpolate(outputs['instances_aux'], size=image.size()[2:],
                                                             mode='bilinear', align_corners=True)

        return outputs

    def prepare_input(self, image):
        prev_mask = None
        dsm=None
        #using_dsm = image.shape[1]==5
        #print("in is_model prepare prev mask, input shape: ", image.shape)
        #print("using dsm!!!!!!!!!!!!!!!!!!!!!!!!!!!!, shape: ", image.shape) if self.use_DSM else print("no dsm used!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!, shape: ", image.shape)

        #if image has 3 channels (RGB) ad empty DSM which is only used if self.use_DSM is true. This is to make the model run with and wihout DSM without crashing
        #if image.shape[1] == 3:
        #    print("image shape before add dummy dsm: ", image.shape)
        #    dummy_dsm = torch.zeros(image.shape[0], 1, image.shape[2], image.shape[3]).to(image.device)
        #    image = torch.cat((image, dummy_dsm), 1)
        #    print("after: ", image.shape)

        if self.use_DSM:
            dsm = image[:,3:4,:,:]
            if self.with_prev_mask:
                prev_mask = image[:,4:,:,:]
                if self.binary_prev_mask:
                    prev_mask = (prev_mask > 0.5).float()
        else:
            if self.with_prev_mask:
                prev_mask = image[:,3:,:,:]
                if self.binary_prev_mask:
                    prev_mask = (prev_mask > 0.5).float()

        
        #adjustment: normalization fkt only works for 3 dimensions, so normalize dsm on its own
        if self.use_DSM:
            dsm = torch.cat((dsm,dsm,dsm), dim=1)
            dsm = self.normalization(dsm)
            dsm = dsm[:,0:1,:,:]
        
        #only ortho (3 channel rgb)
        image = image[:,:3,:,:]



        #normlaize image
        image = self.normalization(image)

        return image, prev_mask, dsm

    def backbone_forward(self, image, coord_features=None):
        raise NotImplementedError

    def get_coord_features(self, image, prev_mask, points):
        if self.clicks_groups is not None: #currently not used
            points_groups = split_points_by_order(points, groups=(2,) + (1, ) * (len(self.clicks_groups) - 2) + (-1,))
            coord_features = [dist_map(image, pg) for dist_map, pg in zip(self.dist_maps, points_groups)]
            coord_features = torch.cat(coord_features, dim=1)
        else: #used
            coord_features = self.dist_maps(image, points)

        if prev_mask is not None:
            coord_features = torch.cat((prev_mask, coord_features), dim=1)

        return coord_features


def split_points_by_order(tpoints: torch.Tensor, groups):
    points = tpoints.cpu().numpy()
    num_groups = len(groups)
    bs = points.shape[0]
    num_points = points.shape[1] // 2

    groups = [x if x > 0 else num_points for x in groups]
    group_points = [np.full((bs, 2 * x, 3), -1, dtype=np.float32)
                    for x in groups]

    last_point_indx_group = np.zeros((bs, num_groups, 2), dtype=np.int)
    for group_indx, group_size in enumerate(groups):
        last_point_indx_group[:, group_indx, 1] = group_size

    for bindx in range(bs):
        for pindx in range(2 * num_points):
            point = points[bindx, pindx, :]
            group_id = int(point[2])
            if group_id < 0:
                continue

            is_negative = int(pindx >= num_points)
            if group_id >= num_groups or (group_id == 0 and is_negative):  # disable negative first click
                group_id = num_groups - 1

            new_point_indx = last_point_indx_group[bindx, group_id, is_negative]
            last_point_indx_group[bindx, group_id, is_negative] += 1

            group_points[group_id][bindx, new_point_indx, :] = point

    group_points = [torch.tensor(x, dtype=tpoints.dtype, device=tpoints.device)
                    for x in group_points]

    return group_points
