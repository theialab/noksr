import torch
import torch.nn as nn
import pytorch_lightning as pl
import importlib
from noksr.model.module import PointTransformerV3
from noksr.utils.samples import BatchedSampler
from noksr.model.general_model import GeneralModel
from torch.nn import functional as F

class noksr(GeneralModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.save_hyperparameters(cfg)
        
        self.latent_dim = cfg.model.network.latent_dim
        self.decoder_type = cfg.model.network.sdf_decoder.decoder_type
        self.eikonal = cfg.data.supervision.eikonal.loss
        self.laplacian = cfg.data.supervision.laplacian.loss
        self.surface_normal_supervision = cfg.data.supervision.on_surface.normal_loss
        self.flip_eikonal = cfg.data.supervision.eikonal.flip
        self.backbone = cfg.model.network.backbone
        if self.backbone == "PointTransformerV3":
            self.point_transformer = PointTransformerV3(
                backbone_cfg=cfg.model.network.point_transformerv3
            )

        module = importlib.import_module('noksr.model.module')
        decoder_class = getattr(module, self.decoder_type)
        self.sdf_decoder = decoder_class(
            decoder_cfg=cfg.model.network.sdf_decoder,
            supervision = 'SDF',
            latent_dim=cfg.model.network.latent_dim,
            feature_dim=cfg.model.network.sdf_decoder.feature_dim,
            hidden_dim=cfg.model.network.sdf_decoder.hidden_dim,
            out_dim=1,
            voxel_size=cfg.data.voxel_size,
            activation=cfg.model.network.sdf_decoder.activation
        )

        # if self.hparams.model.network.mask_decoder.distance_mask:
        if cfg.data.supervision.udf.weight > 0:
            self.mask_decoder = decoder_class(
                decoder_cfg=cfg.model.network.mask_decoder,
            supervision = 'Distance',
            latent_dim=cfg.model.network.latent_dim,
            feature_dim=cfg.model.network.mask_decoder.feature_dim,
            hidden_dim=cfg.model.network.mask_decoder.hidden_dim,
            out_dim=1,
            voxel_size=cfg.data.voxel_size,
            activation=cfg.model.network.mask_decoder.activation
        )

        self.batched_sampler = BatchedSampler(cfg)  # Instantiate the batched sampler with the configurations

    def forward(self, data_dict):
        outputs = {}
        """ Get query samples and ground truth values """
        query_xyz, query_gt_sdf = self.batched_sampler.batch_sdf_sample(data_dict)
        on_surface_xyz, gt_on_surface_normal = self.batched_sampler.batch_on_surface_sample(data_dict)
        if self.hparams.data.supervision.udf.weight > 0:
            mask_query_xyz, mask_query_gt_udf = self.batched_sampler.batch_udf_sample(data_dict)
            outputs['gt_distances'] = mask_query_gt_udf

        outputs['gt_values'] = query_gt_sdf
        outputs['gt_on_surface_normal'] = gt_on_surface_normal

        if self.backbone == "PointTransformerV3":
            pt_data = {}
            pt_data['feat'] = data_dict['point_features']
            pt_data['offset'] = torch.cumsum(data_dict['xyz_splits'], dim=0)
            pt_data['grid_size'] = 0.01
            pt_data['coord'] = data_dict['xyz']
            encoder_output = self.point_transformer(pt_data)

        outputs['values'], *_ = self.sdf_decoder(encoder_output, query_xyz)
        outputs['surface_values'], *_ = self.sdf_decoder(encoder_output, on_surface_xyz)

        if self.eikonal:
            if self.hparams.model.network.grad_type == "Numerical":
                interval = 0.01 * self.hparams.data.voxel_size
                grad_value = []
                for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                    offset_tensor = torch.tensor(offset, device=self.device)[None, :]
                    res_p, *_ = self.sdf_decoder(encoder_output, query_xyz + offset_tensor)
                    res_n, *_ = self.sdf_decoder(encoder_output, query_xyz - offset_tensor)
                    grad_value.append((res_p - res_n) / (2 * interval))
                outputs['pd_grad'] = torch.stack(grad_value, dim=1)
            else:
                xyz = torch.clone(query_xyz)
                xyz.requires_grad = True
                with torch.enable_grad():
                    res, *_ = self.sdf_decoder(encoder_output, xyz)
                    outputs['pd_grad'] = torch.autograd.grad(res, [xyz],
                                                        grad_outputs=torch.ones_like(res),
                                                        create_graph=self.sdf_decoder.training, allow_unused=True)[0]

        if self.laplacian:
            interval = 1.0 * self.hparams.data.voxel_size
            laplacian_value = 0
            
            for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                offset_tensor = torch.tensor(offset, device=self.device)[None, :]
                
                # Calculate numerical gradient
                res, *_ = self.sdf_decoder(encoder_output, query_xyz)
                res_p, *_ = self.sdf_decoder(encoder_output, query_xyz + offset_tensor)
                res_pp, *_ = self.sdf_decoder(encoder_output, query_xyz + 2 * offset_tensor)
                laplacian_value += (res_pp - 2 * res_p + res) / (interval ** 2)
            outputs['pd_laplacian'] = laplacian_value

        if self.surface_normal_supervision:
            if self.hparams.model.network.grad_type == "Numerical":
                interval = 0.01 * self.hparams.data.voxel_size
                grad_value = []
                for offset in [(interval, 0, 0), (0, interval, 0), (0, 0, interval)]:
                    offset_tensor = torch.tensor(offset, device=self.device)[None, :]
                    res_p, *_ = self.sdf_decoder(encoder_output, on_surface_xyz + offset_tensor)
                    res_n, *_ = self.sdf_decoder(encoder_output, on_surface_xyz - offset_tensor)
                    grad_value.append((res_p - res_n) / (2 * interval))
                outputs['pd_surface_grad'] = torch.stack(grad_value, dim=1)
            else:
                xyz = torch.clone(on_surface_xyz)
                xyz.requires_grad = True
                with torch.enable_grad():
                    res, *_ = self.sdf_decoder(encoder_output, xyz)
                    outputs['pd_surface_grad'] = torch.autograd.grad(res, [xyz],
                                                        grad_outputs=torch.ones_like(res),
                                                        create_graph=self.sdf_decoder.training, allow_unused=True)[0]
                                        
        if self.hparams.data.supervision.udf.weight > 0:
            outputs['distances'], *_ = self.mask_decoder(encoder_output, mask_query_xyz)

        return outputs, encoder_output
    
    def loss(self, data_dict, outputs, encoder_output):
        l1_loss = torch.nn.L1Loss(reduction='mean')(torch.clamp(outputs['values'], min = -self.hparams.data.supervision.sdf.max_dist, max=self.hparams.data.supervision.sdf.max_dist), torch.clamp(outputs['gt_values'], min = -self.hparams.data.supervision.sdf.max_dist, max=self.hparams.data.supervision.sdf.max_dist))
        on_surface_loss = torch.abs(outputs['surface_values']).mean()

        mask_loss = torch.tensor(0.0, device=self.device)
        eikonal_loss = torch.tensor(0.0, device=self.device)
        normal_loss = torch.tensor(0.0, device=self.device)
        laplacian_loss = torch.tensor(0.0, device=self.device)
        
        # Create mask for points within max_dist
        valid_mask = (outputs['gt_values'] >= -self.hparams.data.supervision.sdf.max_dist/2) & (outputs['gt_values'] <= self.hparams.data.supervision.sdf.max_dist/2)
        
        # Eikonal Loss computation
        if self.eikonal:
            norms = torch.norm(outputs['pd_grad'], dim=1)  # Compute the norm over the gradient vectors
            eikonal_loss = ((norms - 1) ** 2)[valid_mask].mean()  # Masked eikonal loss

        if self.laplacian:
            laplacian_loss = torch.abs(outputs['pd_laplacian'])[valid_mask].mean()  # Masked laplacian loss

        if self.surface_normal_supervision:
            normalized_pd_surface_grad = -outputs['pd_surface_grad'] / (torch.linalg.norm(outputs['pd_surface_grad'], dim=-1, keepdim=True) + 1.0e-6)
            normal_loss = 1.0 - torch.sum(normalized_pd_surface_grad * outputs['gt_on_surface_normal'], dim=-1).mean()

        # if self.mask:
        if self.hparams.data.supervision.udf.weight > 0:
            mask_loss = torch.nn.L1Loss(reduction='mean')(torch.clamp(outputs['distances'], max=self.hparams.data.supervision.udf.max_dist), torch.clamp(outputs['gt_distances'], max=self.hparams.data.supervision.udf.max_dist))

        return l1_loss, on_surface_loss, mask_loss, eikonal_loss, normal_loss, laplacian_loss

    def training_step(self, data_dict):
        """ UDF auto-encoder training stage """

        batch_size = self.hparams.data.batch_size
        outputs, encoder_output = self.forward(data_dict)

        l1_loss, on_surface_loss, mask_loss, eikonal_loss, normal_loss, laplacian_loss = self.loss(data_dict, outputs, encoder_output)
        self.log("train/l1_loss", l1_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/on_surface_loss", on_surface_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/mask_loss", mask_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/eikonal_loss", eikonal_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log("train/laplacian_loss", laplacian_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)

        total_loss = l1_loss*self.hparams.data.supervision.sdf.weight + on_surface_loss*self.hparams.data.supervision.on_surface.weight + eikonal_loss*self.hparams.data.supervision.eikonal.weight + mask_loss*self.hparams.data.supervision.udf.weight + normal_loss*self.hparams.data.supervision.on_surface.normal_weight + laplacian_loss*self.hparams.data.supervision.laplacian.weight

        return total_loss

    def validation_step(self, data_dict, idx):
        batch_size = 1
        outputs, encoder_output = self.forward(data_dict)
        l1_loss, on_surface_loss, mask_loss, eikonal_loss, normal_loss, laplacian_loss = self.loss(data_dict, outputs, encoder_output)

        self.log("val/l1_loss",  l1_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/on_surface_loss", on_surface_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/mask_loss", mask_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/eikonal_loss", eikonal_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)
        self.log("val/laplacian_loss", laplacian_loss.float(), on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size, logger=True)