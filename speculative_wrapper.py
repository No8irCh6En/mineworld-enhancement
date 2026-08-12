import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
from torchvision import transforms
from PIL import Image

# Import Models
from train_action_predictor import SEQ_LEN, ActionPredictor, MIN_ACTION_TOKEN_ID
from util.attn_model import AttentionTokenPredictor
from vae import VAE
from util.DepthAnythingWrapper import DepthAnythingWrapper, DEPTH_ANYTHING_TRANSFORM
from mcdataset import MCDataset, Buttons
import time

# Constants matching training
PIX_NUM = 336
ACT_NUM = 11
ACTION_VOCAB_OFFSET = 0 # internal logic uses 0-based, we handle offset at io

class SpeculativeInferenceWrapper:
    def __init__(
        self,
        action_model_path="pred_model/action_predictor_latest.pth",
        draft_model_path="pred_model/best_model.pth",
        vae_config="/data/jjli/workspace/mineworld/checkpoints/vae/config.json",
        vae_ckpt="/data/jjli/workspace/mineworld/checkpoints/vae/vae.ckpt",
        device="cuda"
    ):
        self.device = device
        self.dataset_helper = MCDataset()
        # Force init vocab
        self.dataset_helper.make_action_vocab()
        
        # 1. Load Action Predictor
        self.action_model = ActionPredictor().to(device)
        self.action_model.eval()
        # NOTE: ActionPredictor's GRU accepts any sequence length, no fixed seq_len attribute exists.
        # We use the input's sequence length dynamically in action_pred_func.
        self.max_action_history = 8  # max number of past actions to use for prediction
        if os.path.exists(action_model_path):
            state_dict = torch.load(action_model_path, map_location=device)
            # Handle DataParallel wrap if present
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            print(f"State Dict Keys: {list(state_dict.keys())}")
            self.action_model.load_state_dict(state_dict)
            print(f"Loaded Action Predictor from {action_model_path}")
        else:
            print(f"Warning: Action Predictor checkpoint not found at {action_model_path}")

        # 2. Load Draft Model (AttentionTokenPredictor)
        self.draft_model = AttentionTokenPredictor(feature_dim=512).to(device)
        self.draft_model.eval()
        if os.path.exists(draft_model_path):
            state_dict = torch.load(draft_model_path, map_location=device)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            missing_keys, unexpected_keys = self.draft_model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                print(f"[Warning] Draft checkpoint is missing {len(missing_keys)} keys, e.g. {missing_keys[:8]}")
            if unexpected_keys:
                print(f"[Warning] Draft checkpoint has {len(unexpected_keys)} unexpected keys, e.g. {unexpected_keys[:8]}")
            print(f"Loaded Draft Model from {draft_model_path}")
        # Compile AFTER loading state_dict to avoid _orig_mod. prefix mismatch
        self.draft_model = torch.compile(self.draft_model, mode="reduce-overhead")

        # 3. Load VAE
        self.vae = VAE(vae_config, vae_ckpt)
        self.vae.model.to(device)
        self.vae.eval()

        # 4. Load Depth Model
        self.depth_model = DepthAnythingWrapper(device, (384, 224))
        self.depth_model.eval()
        
        # Transforms for image inputs
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Transform for depth input (if different)
        self.depth_transform = DEPTH_ANYTHING_TRANSFORM

    def action_pred_func(self, action_history):
        """
        Predicts next top-k action candidates based on history.
        Args:
            action_history: Tensor [N, 11] — N past actions (variable length).
                            Tokens should be in global ID space (>=8192) or local (0..70).
                            We assume input is global IDs from diag decoding.
        Returns:
            candidates: [K, 11] Tensor (global IDs)
        """
        k = 5 # num candidates
        
        # Reshape to [num_actions, ACT_NUM] — handles variable-length history
        hist_tensor = action_history.to(self.device).reshape(-1, ACT_NUM)
        
        # Limit to last max_action_history actions if longer
        if hist_tensor.size(0) > self.max_action_history:
            hist_tensor = hist_tensor[-self.max_action_history:]
        
        # 2. Normalize to Local IDs (0-70 range) for ActionPredictor
        # If input > MIN_ACTION_TOKEN_ID, shift it.
        hist_tensor = torch.clamp(hist_tensor - MIN_ACTION_TOKEN_ID, min=0)
        
        # Input shape to model: [B=1, Seq, 11]
        inp = hist_tensor.unsqueeze(0)
        
        # 3. Predict
        # Output: [1, K, 11] (Local IDs)
        with torch.no_grad():
            top_k_seqs = self.action_model.predict_top_k_vectors(inp, k=k)
        
        candidates = top_k_seqs.squeeze(0) # [K, 11]
        
        # 4. Convert back to Global IDs for consistency with Main Model
        return candidates + MIN_ACTION_TOKEN_ID

    def draft_func(self, prev_tokens, action_candidates, merge=True):
        """
        Generates draft image tokens for the next frame with Image-Space Soft Merging.
        
        Args:
            prev_tokens: [336] Tensor (Global Token IDs of previous frame)
            action_candidates: [K, 11] Tensor (Global Token IDs of actions to speculate)
            merge: bool, whether to use confidence-based merging (default: True)
        
        Returns:
            draft_tokens: [K, 336] Tensor (Predicted Image Token IDs)
        """
        # prev_tokens: iterable/list or tensor of token ids length PIX_NUM
        # 返回: tensor of draft token ids (1, PIX_NUM) or (1,14,24)
        device = self.device
        # 1. tokens -> tensor and clamp
        img_tokens_prev = torch.as_tensor(prev_tokens, dtype=torch.long, device=device).view(1, 14, 24)
        img_tokens_prev = torch.clamp(img_tokens_prev, max=8191)
        
        with torch.no_grad():
            # 2. decode tokens -> prev image (uint8 HWC 或 CHW)
            prev_img_uint8 = self.vae.token2image(img_tokens_prev)  # may return np.ndarray or tensor

            # 3. convert -> torch tensor (C,H,W), float [0,1], then to [-1,1]
            if isinstance(prev_img_uint8, np.ndarray):
                prev_img_tensor = torch.from_numpy(prev_img_uint8).to(device)
            else:
                prev_img_tensor = torch.as_tensor(prev_img_uint8).to(device)

            # ensure shape CHW
            if prev_img_tensor.ndim == 3 and prev_img_tensor.shape[2] == 3:
                # HWC -> CHW
                prev_img_tensor = prev_img_tensor.permute(2, 0, 1)
            elif prev_img_tensor.ndim == 3 and prev_img_tensor.shape[0] == 3:
                pass
            else:
                # fallback: try to reshape
                prev_img_tensor = prev_img_tensor.view(3, prev_img_tensor.shape[-2], prev_img_tensor.shape[-1])

            prev_img_tensor = prev_img_tensor.float() / 255.0
            # normalize to [-1,1] as VAE style
            prev_img_tensor = (prev_img_tensor - 0.5) / 0.5
            prev_img_tensor = prev_img_tensor.unsqueeze(0)  # [1,3,H,W]

            # 4. prepare depth input exactly like train_uncertainty:
            #    convert to [0,1], then ImageNet normalize and resize to (H=224, W=384)
            img_01 = (prev_img_tensor * 0.5 + 0.5)  # [0,1]
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            norm_input = (img_01 - mean) / std

            # IMPORTANT: depth model expects (H=224, W=384) -> ensure this order
            depth_h, depth_w = (224, 384)
            current_depth_input = F.interpolate(norm_input, size=(depth_h, 392), mode='bilinear', align_corners=False)

            # 5. compute depth_map, normalize and resize back to VAE image size if needed
            depth_map = self.depth_model(current_depth_input)
            if depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(1)  # (1,1,H,W)
            # normalize depth_map
            d_min, d_max = depth_map.min(), depth_map.max()
            depth_map = (depth_map - d_min) / (d_max - d_min + 1e-6)

            # make sure depth_map spatial matches prev_img_tensor; resize if necessary
            _, _, h_img, w_img = prev_img_tensor.shape
            if depth_map.shape[-2:] != (h_img, w_img):
                depth_map = F.interpolate(depth_map, size=(h_img, w_img), mode='bilinear', align_corners=False)

            # 6. input to draft model: concat channels -> [1,4,H,W]
            img_depth = torch.cat([prev_img_tensor, depth_map], dim=1)
            
            # Expand for K candidates: [K, 4, H, W]
            K = action_candidates.size(0)
            img_depth_expanded = img_depth.expand(K, -1, -1, -1)
            
            # 3. Process Actions
            local_act_tokens = action_candidates.clone()
            local_act_tokens = local_act_tokens - MIN_ACTION_TOKEN_ID
            
            start_time = time.perf_counter()
            
            action_vecs = self._batch_tokens_to_vectors(local_act_tokens).to(self.device)
            
            end_time = time.perf_counter()
            print(f"[DEBUG] Action vector conversion time for {K} candidates: {end_time - start_time:.6f} seconds")
            
            # 4. Predict Draft
            # pred_conf_logits: [K, 1, 14, 24], logits_token: [K, 8192, 14, 24]
            pred_conf_logits, logits_token = self.draft_model(img_depth_expanded, action_vecs) 
            
            # Get Raw Predicted Tokens
            pred_tokens = torch.argmax(logits_token, dim=1) # [K, 14, 24]
            
            # --- Image Space Merge Logic ---
            if merge:
                # 4.1 Decode Predicted Tokens -> Predicted Image (batch, GPU)
                # token2image only accepts a single [1,14,24]; use token2image_gpu
                # which supports a batch [K,14,24] and returns [K,3,H,W] in [-1,1].
                pred_img_gpu = self.vae.token2image_gpu(pred_tokens)  # [K,3,H,W], [-1,1]
                
                # 4.2 Confidence Map (sigmoid) -> upsample to image spatial size
                confidence = torch.sigmoid(pred_conf_logits).squeeze(1)  # [K, 14, 24]
                # get target spatial size from decoded prev image
                _, _, h_img, w_img = img_depth_expanded.shape  # [K,4,H,W]
                conf_img = F.interpolate(confidence.unsqueeze(1), size=(h_img, w_img), mode='bilinear', align_corners=False)  # [K,1,H,W]

                # 4.3 Draft image is already [K,3,H,W] in [-1,1] on GPU
                draft_img_tensor = pred_img_gpu

                # 4.4 Merge: use only RGB channels from img_depth_expanded (first 3 channels)
                prev_rgb = img_depth_expanded[:, :3, :, :]  # [K,3,H,W]
                merged_img_tensor = conf_img * draft_img_tensor + (1.0 - conf_img) * prev_rgb  # broadcasting ok

                # 4.5 Encode Merged Image -> Final Tokens
                final_tokens = self.vae.tokenize_images(merged_img_tensor)
                return final_tokens.view(action_candidates.size(0),PIX_NUM)
            
            # If no merging, fallback to raw draft tokens
            draft_ids = pred_tokens.view(K, PIX_NUM) # [K, 336]
            return draft_ids

    def _batch_tokens_to_vectors(self, token_indices_batch):
        """
        Converts batch of action token indices (0-70 range) to continuous action vectors.
        Output vector order: [fwd, back, left, right, jump, sneak, sprint, camX, camY, atk, use]
        """
        batch_size = token_indices_batch.size(0)
        vecs = torch.zeros((batch_size, 11), dtype=torch.float32)
        
        # Download to CPU for directory lookup (or reimplement logic on GPU if vocab is mapped)
        # Using CPU for flexibility as vocab is dict
        tokens_np = token_indices_batch.cpu().numpy()
        
        # Invert vocab for fast lookup
        # vocab: name -> idx. inv_vocab: idx -> name
        if not hasattr(self, 'inv_vocab'):
             self.inv_vocab = {v: k for k, v in self.dataset_helper.action_vocab.items()}
        
        for b in range(batch_size):
            # Reconstruct dict-like status
            # We map specific positions in the 11-len sequence back to attributes
            # Sequence: [BOS, Cam0(y), Cam1(x), Hotbar, F/B, L/R, Sp/Sn, Use/Atk, Jmp, Pck, EOS]
            row = tokens_np[b]
            
            # 1. Camera (Indices 1 and 2)
            # row[1] is Cam Y tok, row[2] is Cam X tok
            cam_y_name = self.inv_vocab.get(row[1], "")
            cam_x_name = self.inv_vocab.get(row[2], "")
            
            # Extract bin index from string "cam_0_5" -> 5
            c_y = int(cam_y_name.split('_')[-1]) if "cam" in cam_y_name else 0 # usually center
            c_x = int(cam_x_name.split('_')[-1]) if "cam" in cam_x_name else 0
            
            cam_val = self.dataset_helper.camera_quantizer.undiscretize(np.array([c_y, c_x]))
            vecs[b, 7] = cam_val[0] # Cam Y (Pitch) -> Index 7
            vecs[b, 8] = cam_val[1] # Cam X (Yaw)   -> Index 8
            
            # 2. Forward/Back (Index 4 in token seq)
            fb_token = row[4]
            name = self.inv_vocab.get(fb_token, "")
            if name == "forward": vecs[b, 0] = 1.0
            elif name == "back":  vecs[b, 1] = 1.0
            
            # 3. Left/Right (Index 5)
            lr_token = row[5]
            name = self.inv_vocab.get(lr_token, "")
            if name == "left": vecs[b, 2] = 1.0
            elif name == "right": vecs[b, 3] = 1.0
            
            # 4. Jump (Index 8)
            jmp_token = row[8]
            name = self.inv_vocab.get(jmp_token, "")
            if name == "jump": vecs[b, 4] = 1.0
            
            # 5. Sneak/Sprint (Index 6)
            ss_token = row[6]
            name = self.inv_vocab.get(ss_token, "")
            if name == "sneak": vecs[b, 5] = 1.0
            elif name == "sprint": vecs[b, 6] = 1.0
            
            # 6. Attack/Use (Index 7)
            au_token = row[7]
            name = self.inv_vocab.get(au_token, "")
            if name == "attack": vecs[b, 9] = 1.0
            elif name == "use": vecs[b, 10] = 1.0
            
        return vecs

# Helper function to instantiate easily in your main script
def get_inference_functions(
    action_model_path="pred_action/action_predictor_latest.pth", 
    draft_model_path="pred_model_uncertainty/best_model.pth"
):
    """
    Returns (draft_func, action_pred_func) ready for diagonal decoding loop.
    """
    wrapper = SpeculativeInferenceWrapper(
        action_model_path=action_model_path,
        draft_model_path=draft_model_path
    )
    return wrapper.draft_func, wrapper.action_pred_func

def draft_func_subs(**kwargs):
    bsz = 5
    confidence = torch.rand(bsz, 1, 14, 24)
    logits = torch.zeros(bsz, 8192, 14, 24)
    logits[:, 20, :, :] = 1.0
    return confidence, logits