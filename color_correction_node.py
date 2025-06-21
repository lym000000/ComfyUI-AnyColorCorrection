import numpy as np
import cv2
import torch

class AnyColorCorrectionNode:
    """
    A ComfyUI custom node to correct colors in a masked region of a target image
    based on a reference image — useful for clothing, backgrounds, or any region.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
            },
            "optional": {
                "mask_ref": ("MASK",),
                "mask_target": ("MASK",),
                "color_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preserve_luminosity": ("BOOLEAN", {"default": True}),
                "method": (["statistical", "histogram", "lab_transfer"], {"default": "lab_transfer"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "correct_colors"
    CATEGORY = "image/color_correction"
    DESCRIPTION = "Color corrects a masked region in the target image based on the reference image."

    def correct_colors(self, image_ref, image_target, mask_ref=None, mask_target=None,
                       color_threshold=0.15, strength=0.8, preserve_luminosity=True, method="lab_transfer"):
    
        device = image_ref.device
    
        # Normalize batch dimensions
        if image_ref.dim() == 3 and image_target.dim() == 4:
            ref_batch = image_ref.unsqueeze(0).repeat(image_target.size(0), 1, 1, 1)
            tgt_batch = image_target
        elif image_ref.dim() == 4 and image_target.dim() == 3:
            ref_batch = image_ref
            tgt_batch = image_target.unsqueeze(0).repeat(image_ref.size(0), 1, 1, 1)
        elif image_ref.dim() == 3 and image_target.dim() == 3:
            ref_batch = image_ref.unsqueeze(0)
            tgt_batch = image_target.unsqueeze(0)
        else:
            ref_batch = image_ref
            tgt_batch = image_target
    
        batch_size = tgt_batch.size(0)
    
        # Ensure ref_batch matches tgt_batch in size
        if ref_batch.size(0) == 1 and batch_size > 1:
            ref_batch = ref_batch.repeat(batch_size, 1, 1, 1)
        elif ref_batch.size(0) != batch_size:
            raise ValueError(f"Mismatched batch sizes: ref_batch={ref_batch.size(0)}, tgt_batch={batch_size}")
    
        # Handle mask broadcasting
        if mask_ref is not None and mask_ref.dim() == 3:
            mask_ref = mask_ref.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        if mask_target is not None and mask_target.dim() == 3:
            mask_target = mask_target.unsqueeze(0).repeat(batch_size, 1, 1, 1)
    
        results = []
    
        for b in range(batch_size):
            ref = ref_batch[b].cpu().numpy()
            tgt = tgt_batch[b].cpu().numpy()
    
            h, w, _ = ref.shape
            mask_r = mask_ref[b].cpu().numpy() if mask_ref is not None else np.ones((h, w), dtype=np.uint8) * 255
            mask_t = mask_target[b].cpu().numpy() if mask_target is not None else np.ones((h, w), dtype=np.uint8) * 255
    
            ref_u8 = (np.clip(ref * 255.0, 0, 255)).astype(np.uint8)
            tgt_u8 = (np.clip(tgt * 255.0, 0, 255)).astype(np.uint8)
    
            mask_r_bin = ((mask_r > 0.5).astype(np.uint8)) * 255
            mask_t_bin = ((mask_t > 0.5).astype(np.uint8)) * 255
    
            ref_stats = self._get_masked_color_stats(ref_u8, mask_r_bin)
            tgt_stats = self._get_masked_color_stats(tgt_u8, mask_t_bin)
    
            color_diff = self._calculate_color_difference(ref_stats, tgt_stats)
            result_img = tgt_u8.copy()
    
            if color_diff > color_threshold:
                if method == "statistical":
                    corrected = self._match_colors_statistical(tgt_u8, mask_t_bin, ref_stats, tgt_stats, strength, preserve_luminosity)
                elif method == "histogram":
                    corrected = self._match_colors_histogram(tgt_u8, ref_u8, mask_t_bin, strength)
                elif method == "lab_transfer":
                    corrected = self._match_colors_lab(tgt_u8, ref_u8, mask_t_bin, mask_r_bin, strength, preserve_luminosity)
                else:
                    corrected = tgt_u8
    
                mask_3ch = cv2.merge([mask_t_bin] * 3)
                mask_inv = cv2.bitwise_not(mask_3ch)
                bg = cv2.bitwise_and(result_img, mask_inv)
                fg = cv2.bitwise_and(corrected, mask_3ch)
                result_img = cv2.add(bg, fg)
    
            result_float = result_img.astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_float))
    
        result_tensor = torch.stack(results).to(device)
        return (result_tensor if result_tensor.size(0) > 1 else result_tensor[0],)


    def _get_masked_color_stats(self, image, mask):
        if len(mask.shape) > 2:
            mask = mask[:, :, 0]
        y, x = np.where(mask > 0)
        if len(y) == 0:
            return {'mean': np.zeros(3), 'std': np.zeros(3)}
        pixels = image[y, x]
        return {'mean': pixels.mean(axis=0), 'std': pixels.std(axis=0)}

    def _calculate_color_difference(self, stats1, stats2):
        mean_diff = np.linalg.norm(stats1['mean'] - stats2['mean']) / (255 * np.sqrt(3))
        std_diff = np.linalg.norm(stats1['std'] - stats2['std']) / (255 * np.sqrt(3))
        return 0.7 * mean_diff + 0.3 * std_diff

    def _match_colors_statistical(self, target_img, target_mask, source_stats, target_stats, strength, preserve_luminosity):
        result = target_img.copy()
        if preserve_luminosity:
            result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
            for i in range(1, 3):
                ms, mt = source_stats['mean'][i], target_stats['mean'][i]
                ss, st = source_stats['std'][i], target_stats['std'][i]
                if st > 0:
                    result_lab[:, :, i] = ((result_lab[:, :, i] - mt) * (ss / st) * strength + mt * (1 - strength) + ms * strength).clip(0, 255)
            result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        else:
            for i in range(3):
                ms, mt = source_stats['mean'][i], target_stats['mean'][i]
                ss, st = source_stats['std'][i], target_stats['std'][i]
                if st > 0:
                    result[:, :, i] = ((result[:, :, i] - mt) * (ss / st) * strength + mt * (1 - strength) + ms * strength).clip(0, 255)
        return result

    def _match_colors_histogram(self, target_img, source_img, target_mask, strength):
        target_hsv = cv2.cvtColor(target_img, cv2.COLOR_RGB2HSV)
        source_hsv = cv2.cvtColor(source_img, cv2.COLOR_RGB2HSV)
        result_hsv = target_hsv.copy()
        for i in range(3):
            source_hist, _ = np.histogram(source_hsv[:, :, i].flatten(), 256, [0, 256])
            target_hist, _ = np.histogram(target_hsv[:, :, i].flatten(), 256, [0, 256])
            source_cdf = source_hist.cumsum() / source_hist.sum()
            target_cdf = target_hist.cumsum() / target_hist.sum()
            mapping = np.zeros(256, dtype=np.uint8)
            for j in range(256):
                mapping[j] = np.argmin(np.abs(source_cdf - target_cdf[j]))
            matched = mapping[target_hsv[:, :, i]]
            result_hsv[:, :, i] = (target_hsv[:, :, i] * (1 - strength) + matched * strength).astype(np.uint8)
        return cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)

    def _match_colors_lab(self, target_img, source_img, target_mask, source_mask, strength, preserve_luminosity):
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
        result_lab = target_lab.copy()
        y_t, x_t = np.where(target_mask > 0)
        y_s, x_s = np.where(source_mask > 0)
        if len(y_t) == 0 or len(y_s) == 0:
            return target_img.copy()
        target_vals = target_lab[y_t, x_t]
        source_vals = source_lab[y_s, x_s]
        target_means = np.mean(target_vals, axis=0)
        target_stds = np.std(target_vals, axis=0)
        source_means = np.mean(source_vals, axis=0)
        source_stds = np.std(source_vals, axis=0)
        for i in range(3):
            if i == 0 and preserve_luminosity:
                continue
            if target_stds[i] > 0:
                factor = (source_stds[i] / target_stds[i]) * strength
                offset = source_means[i] * strength + target_means[i] * (1 - strength) - target_means[i] * factor
                channel = result_lab[:, :, i].astype(np.float32)
                result_lab[:, :, i] = np.clip(channel * factor + offset, 0, 255).astype(np.uint8)
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)

NODE_CLASS_MAPPINGS = {
    "AnyColorCorrection": AnyColorCorrectionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyColorCorrection": "Any Color Correction"
}
