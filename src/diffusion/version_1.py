from einops import rearrange, reduce, repeat
import torch.nn.functional as F
import torch
import gc
from src.utils import *
from src.flow_utils import get_mapping_ind, warp_tensor
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import AttnProcessor2_0
from typing import Any, Dict, List, Optional, Tuple, Union
import sys
sys.path.append("./src/ebsynth/deps/gmflow/")
from gmflow.geometry import flow_warp, forward_backward_consistency_check

class FRESCOAttnProcessor2_0:
    """
    Hack self attention to FRESCO-based attention
    * adding spatial-guided attention
    * adding temporal-guided attention
    * adding cross-frame attention
    
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    Usage
    frescoProc = FRESCOAttnProcessor2_0(2, attn_mask)
    attnProc = AttnProcessor2_0()
    
    attn_processor_dict = {}
    for k in pipe.unet.attn_processors.keys():
        if k.startswith("up_blocks.2") or k.startswith("up_blocks.3"):
            attn_processor_dict[k] = frescoProc
        else:
            attn_processor_dict[k] = attnProc
    pipe.unet.set_attn_processor(attn_processor_dict)
    """

    def __init__(self, unet_chunk_size=2, controller=None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.unet_chunk_size = unet_chunk_size
        self.controller = controller
            
    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
            #print('187.hidden_states.shape: batch_size, channel, height * width=', batch_size, channel, height * width)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        #print('204.query.shape: ', query.shape)
        #print('204.hidden_states.shape: ',hidden_states.shape)

        crossattn = False
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
            if self.controller and self.controller.store:
                self.controller(hidden_states.detach().clone())
        else:
            crossattn = True
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            
        # BC * HW * 8D
        key = attn.to_k(encoder_hidden_states) 
        #print('218.key.shape: ', key.shape)
        value = attn.to_v(encoder_hidden_states)
        #print('220.value.shape: ', value.shape)
        
        query_raw, key_raw = None, None
        if self.controller and self.controller.use_interattn and (not crossattn):
            query_raw, key_raw = query.clone(), key.clone()

        inner_dim = key.shape[-1] # 8D
        head_dim = inner_dim // attn.heads # D

        print("self.controller.use_cfattn: ",self.controller.use_cfattn)
        print("crossattn: ",crossattn)
        
        '''for efficient cross-frame attention'''
        if self.controller and self.controller.use_cfattn and (not crossattn):
            video_length = key.size()[0] // self.unet_chunk_size
            former_frame_index = [0] * video_length
            attn_mask = []
            bwd_flows = []
            
            if self.controller.attn_mask is not None:
                for m in self.controller.attn_mask:
                    for n in m: 
                        if n.shape[1]==key.shape[1]:
                            attn_mask.append(n)
            attn_mask=torch.stack(attn_mask,dim=0)
            
            query=rearrange(query,"(b f) d c -> b f d c",f=video_length)
            key=rearrange(key,"(b f) d c -> b f d c",f=video_length)
            value=rearrange(value,"(b f) d c -> b f d c",f=video_length)
            
            hidden_states = torch.zeros(query.shape[0],query.shape[1],query.shape[2],query.shape[3], dtype=torch.float16).to(query.device)
            for i in range(attn_mask.shape[0]):
                query_i = query[:, attn_mask[i]]
                #print("247.query_i.shape: ",query_i.shape)
                #query_i_q = attn.to_q(query_i) 
                #query_i_q = query_i_q.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
                query_i = query_i.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
                key_i = key[:,i].view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
                #print("250.key_i.shape: ",key_i.shape)
                value_i = value[:,i].view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)  
                #query_k_i=query[:,i]
                #query_k_i=attn.to_k(query_k_i)
                #query_k_i = query_k_i.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
                #query_v_i = query[:,i].view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)  
                
                #query_i = F.scaled_dot_product_attention(
                #    query_i, query_k_i, query_v_i, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                #)         
                query_i = F.scaled_dot_product_attention(
                    query_i, key_i, value_i, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
                )
                query_i = query_i.transpose(1, 2).contiguous().view(query.shape[0], -1, attn.heads * head_dim)
                hidden_states[:,attn_mask[i]]=query_i
            hidden_states = rearrange(hidden_states,"b f d c -> (b f) d c")
            hidden_states = hidden_states.view(batch_size,-1,attn.heads,head_dim).transpose(1,2)
            
        else:
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        
        # BC * 8 * HW * D --> BC * HW * 8D
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype) 

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

@torch.no_grad()
def get_flow_and_interframe_paras(flow_model, imgs, visualize_pipeline=False):
    """
    Get parameters for temporal-guided attention and optimization
    * predict optical flow and occlusion mask
    * compute pixel index correspondence for FLATTEN
    """
    images = torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs], dim=0).cuda()
    imgs_torch = torch.cat([numpy2tensor(img) for img in imgs], dim=0)

    mask=[]
    fwd_flows_result=[]
    bwd_flows_result=[]
    bwd_occs_result=[]
    fwd_occs_result=[]
    for i in range(1,len(images)):
        duplicated_images = images[i-1].unsqueeze(0).repeat(len(images)-i, 1, 1, 1)
        print("965.duplicated_images.shape: ",duplicated_images.shape)
        reshuffle_list = list(range(i,len(images)))    
        print("967.images[reshuffle_list].shape: ",images[reshuffle_list].shape)
        results_dict = flow_model(duplicated_images, images[reshuffle_list], attn_splits_list=[2], 
                                corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True)
        flow_pr = results_dict['flow_preds'][-1]  # [2*B, 2, H, W]
        fwd_flows, bwd_flows = flow_pr.chunk(2)   # [B, 2, H, W]
        fwd_flows_result.append(fwd_flows[0])
    
        bwd_flows_result.append(bwd_flows[0])
        
        fwd_occs, bwd_occs = forward_backward_consistency_check(fwd_flows, bwd_flows) # [B, H, W]
        warped_image1 = flow_warp(duplicated_images, bwd_flows)
        bwd_occs = torch.clamp(bwd_occs + (abs(images[reshuffle_list]-warped_image1).mean(dim=1)>255*0.25).float(), 0 ,1)
        bwd_occs_result.append(bwd_occs[0])
        
        warped_image2 = flow_warp(images[reshuffle_list], fwd_flows)
        fwd_occs = torch.clamp(fwd_occs + (abs(duplicated_images-warped_image2).mean(dim=1)>255*0.25).float(), 0 ,1)    
        fwd_occs_result.append(fwd_occs[0])
        if visualize_pipeline:
            print('visualized occlusion masks based on optical flows')
            viz = torchvision.utils.make_grid(imgs_torch * (1-fwd_occs.unsqueeze(1)), len(images), 1)
            visualize(viz.cpu(), 90)
            viz = torchvision.utils.make_grid(imgs_torch[reshuffle_list] * (1-bwd_occs.unsqueeze(1)), len(images), 1)
            visualize(viz.cpu(), 90) 
        
        attn_mask = []
        attn_mask_last=[]

        for scale in [8.0, 16.0, 32.0]:
            bwd_occs_ = F.interpolate(bwd_occs.unsqueeze(1), scale_factor=1./scale, mode='bilinear') #[f-i,1,h,w]
            print("990.bwd_occs_.shape: ",bwd_occs_.shape)
            #attn_mask_tmp = bwd_occs_.reshape(bwd_occs_.shape[0],-1)<0.5
            if i > 1:
                attn_mask_ = torch.cat(((bwd_occs_[0:1].reshape(1,-1)>-1), bwd_occs_.reshape(bwd_occs_.shape[0],-1)<0.5), dim=0)
                if i==len(images)-1:
                    attn_mask_last_ = bwd_occs_[0:1].reshape(1,-1)>-1
                for j in range(1,i):
                    for m in mask[j-1]:# [f-i,h,w]
                        if m.shape[1]==attn_mask_.shape[1]:
                            attn_mask_=attn_mask_ &~m[-attn_mask_.shape[0]:]
                            if i==len(images)-1:
                                attn_mask_last_=attn_mask_last_ &~m[-attn_mask_last_.shape[0]:,:]
                attn_mask_=torch.cat(((bwd_occs_[0:1].reshape(1,-1)>2).repeat(len(images)-attn_mask_.shape[0],1), attn_mask_), dim=0)
                if i==len(images)-1:
                    attn_mask_last_=attn_mask_last_ &~attn_mask_[-attn_mask_last_.shape[0]:,:]
                    attn_mask_last_=torch.cat(((bwd_occs_[0:1].reshape(1,-1)>2).repeat(len(images)-attn_mask_last_.shape[0],1), attn_mask_last_), dim=0)
            else:
                attn_mask_ = torch.cat(((bwd_occs_[0:1].reshape(1,-1)>-1), bwd_occs_.reshape(bwd_occs_.shape[0],-1)<0.5), dim=0)

            attn_mask += [attn_mask_]
            if i==len(images)-1:
                attn_mask_last+=[attn_mask_last_]
        mask.append(attn_mask)
    mask.append(attn_mask_last)
    results_dict_2 = flow_model(images[-1].unsqueeze(0), images[0].unsqueeze(0), attn_splits_list=[2], 
                               corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True)
    flow_pr_2 = results_dict_2['flow_preds'][-1]  # [2*B, 2, H, W]
    fwd_flows_2, bwd_flows_2 = flow_pr_2.chunk(2)   # [B, 2, H, W]
    fwd_flows_result.append(fwd_flows_2.squeeze(0))
    
    bwd_flows_result.append(bwd_flows_2.squeeze(0))
    fwd_occs_2, bwd_occs_2 = forward_backward_consistency_check(fwd_flows_2, bwd_flows_2) # [B, H, W]
    warped_image1 = flow_warp(images[-1].unsqueeze(0), bwd_flows_2)
    bwd_occs_2 = torch.clamp(bwd_occs_2 + (abs(images[0].unsqueeze(0)-warped_image1).mean(dim=1)>255*0.25).float(), 0 ,1)
    bwd_occs_result.append(bwd_occs_2.squeeze(0))
        
    warped_image2 = flow_warp(images[0].unsqueeze(0), fwd_flows_2)
    fwd_occs_2 = torch.clamp(fwd_occs_2 + (abs(images[-1].unsqueeze(0)-warped_image2).mean(dim=1)>255*0.25).float(), 0 ,1)    
    fwd_occs_result.append(fwd_occs_2.squeeze(0))
    
    fwd_flows_result = torch.stack(fwd_flows_result, dim=0)
    bwd_flows_result = torch.stack(bwd_flows_result, dim=0)
    print("1027.fwd_flows_result.shape: ",fwd_flows_result.shape)
    bwd_occs_result = torch.stack(bwd_occs_result, dim=0)
    fwd_occs_result = torch.stack(fwd_occs_result, dim=0)
    print("1030.fwd_occs_result.shape: ",fwd_occs_result.shape)
    #mask = torch.stack([torch.stack(sublist, dim=0) for sublist in mask], dim=0)
    #mask = mask.transpose(0, 1)
    #print("1033.mask.shape: ",mask.shape)
    gc.collect()
    torch.cuda.empty_cache()
    
    return [fwd_flows_result, bwd_flows_result], [fwd_occs_result, bwd_occs_result], mask
    