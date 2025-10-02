"""
Класс хранения данных об загруженных и обработанных исследованиях
"""
import numpy as np
import nibabel as nib
import torch
from scipy.ndimage import zoom as ndi_zoom, binary_erosion, generate_binary_structure
from scipy.ndimage import label as ndi_label
from skimage.exposure import equalize_adapthist
from totalsegmentator.python_api import totalsegmentator


class ScreenerModel:
    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.model = torch.load(ckpt_path, map_location="cpu", weights_only=False).to(device)
        self.model.eval()
        
        self.total_dict = {
        10: "lung_upper_lobe_left",
        11: "lung_lower_lobe_left",
        12: "lung_upper_lobe_right",
        13: "lung_middle_lobe_right",
        14: "lung_lower_lobe_right",
        15: "esophagus",
        16: "trachea",
        51: "heart",
        52: "aorta",
        92: "rib_left_1",
        93: "rib_left_2",
        94: "rib_left_3",
        95: "rib_left_4",
        96: "rib_left_5",
        97: "rib_left_6",
        98: "rib_left_7",
        99: "rib_left_8",
        100: "rib_left_9",
        101: "rib_left_10",
        102: "rib_left_11",
        103: "rib_left_12",
        104: "rib_right_1",
        105: "rib_right_2",
        106: "rib_right_3",
        107: "rib_right_4",
        108: "rib_right_5",
        109: "rib_right_6",
        110: "rib_right_7",
        111: "rib_right_8",
        112: "rib_right_9",
        113: "rib_right_10",
        114: "rib_right_11",
        115: "rib_right_12",
        }
        
        self.total_dict_names = {value: key for key, value in self.total_dict.items()}

    def _voxel_spacing(self, affine: np.ndarray):
        return tuple(np.abs(np.diag(affine[:3, :3])))

    def _resample_iso(self, array: np.ndarray, spacing, target_spacing=(1.0, 1.0, 1.0), order=1):
        factors = tuple(s / t for s, t in zip(spacing, target_spacing))
        return ndi_zoom(array, zoom=factors, order=order)

    def _normalize(self, x: np.ndarray):
        x = x.astype(np.float32)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = equalize_adapthist(x, clip_limit=0.05).astype(np.float32)
        return x

    def _boundary_from_binary(self, m: np.ndarray):
        er = binary_erosion(m, structure=np.ones((3,3,3)))
        return (m & (~er)).astype(np.uint8)
    
    
    VOXEL_SPACING = (1.0, 1.0, 1.0)   # target spacing
    MIN_IMAGE_SIZE = (96, 96, 96)     # after crop, before resample

    # ---- helpers mirrored from prescreener.py ----
    def _affine_to_voxel_spacing(self, affine: np.ndarray):
        return tuple(np.abs(np.diag(affine[:3, :3])))

    def _is_diagonal(self, M: np.ndarray) -> bool:
        return np.allclose(M, np.diag(np.diag(M)))

    def _to_canonical_orientation(self, image: np.ndarray, voxel_spacing, affine: np.ndarray):
        """
        Match prescreener behavior:
        - If affine[:3,:3] diagonal, flip axes with negative signs.
        - Else (non-orthogonal), leave as-is (the prescreener returns None/None; here we just skip flips).
        """
        A = affine[:3, :3]
        if not self._is_diagonal(A):
            return image, voxel_spacing  # conservative: skip if non-diagonal

        # flip axes where diagonal element is negative
        flip_axes = tuple(np.where(np.diag(A) < 0)[0].tolist())
        if len(flip_axes):
            image = np.flip(image, axis=flip_axes)
        
        image = image.transpose((1, 0, 2))
        image = np.flip(image, axis=(0, 1, 2))
        image = image.copy()
        
        return image, voxel_spacing

    def _mask_to_bbox(self, mask: np.ndarray) -> np.ndarray:
        """
        Smallest [start, stop) box containing mask==True.
        Equivalent to prescreener's iterative projection approach.
        """
        if not mask.any():
            raise ValueError("The mask is empty.")
        idx = np.where(mask)
        start = [int(idx[d].min()) for d in range(mask.ndim)]
        stop  = [int(idx[d].max()) + 1 for d in range(mask.ndim)]
        return np.array([start, stop], dtype=int)

    def _crop_to_box(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        (start, stop) = box
        sl = tuple(slice(start[i], stop[i]) for i in range(3))
        return image[sl]

    def _resample_iso(self, image: np.ndarray, in_spacing, out_spacing=(1.0, 1.0, 1.0), order=1):
        factors = tuple(s / t for s, t in zip(in_spacing, out_spacing))
        return ndi_zoom(image, zoom=factors, order=order)

    def _normalize(self, x: np.ndarray):
        x = x.astype(np.float32)
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        x = equalize_adapthist(x, clip_limit=0.05).astype(np.float32)
        return x
    
    
    def _segment(self, nii_file):
        roi_subset = list(self.total_dict.values())
        
        return totalsegmentator(
                    input = nii_file, 
                    output = None, 
                    fast=True, 
                    ml=True, 
                    device="gpu",
                    roi_subset=roi_subset,
                    preview=False,
                    skip_saving=False # Не пропускать сохранение, если сегментация удалась
                )

    @torch.no_grad()
    def predict(self, nii, study_id = None, series_id = None, class_threshold = 0.7,
               class_sum = 1800, seg_threshold = 0.6):
        

        result = {
                'study_uid':study_id,
                'series_uid': series_id,
                'probability_of_pathology': None,
                'pathology': None,
                'processing_status': "",
                'time_of_processing' : None,
                'most_dangerous_pathology_type': "",
                'pathology_localization': None,
                'debug_message': ""
                }
        try:
        #if True:
            segment_nii = self._segment(nii)
            segmentation = segment_nii.get_fdata().astype(np.float32)
            print('Segmented shape:', segmentation.shape)
        
            for name in ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_upper_lobe_right",
            "lung_middle_lobe_right", "lung_lower_lobe_right", "trachea", "heart", "aorta"]:
                if np.sum(segmentation==self.total_dict_names[name])<10000:
                    result['processing_status'] = 'Failure' 
                    result['debug_message'] = name + 'not found in the CT' 
                    return result, None, None, None, None        

            image = nii.get_fdata().astype(np.float32)
            orig_shape = image.shape
            orig_affine = nii.affine

            orig_spacing = self._affine_to_voxel_spacing(nii.affine)
            image, orig_spacing = self._to_canonical_orientation(image, orig_spacing, nii.affine)
        

            # size guard (pre-resample, as in the file)
            if any(image.shape[i] < self.MIN_IMAGE_SIZE[i] for i in range(3)):
                result['processing_status'] = 'Failure' 
                result['debug_message'] = 'Image too small ' + str(image.shape)
                return result, segment_nii, None, None, None

            # resample to 1×1×1 mm (linear)
            iso_img = self._resample_iso(image, in_spacing=orig_spacing, out_spacing=self.VOXEL_SPACING, order=1)
            iso_img[iso_img == 0] = -1024.0

            # tight crop to foreground (values > global min), as in prescreener
            box = self._mask_to_bbox(iso_img > iso_img.min())
            iso_img = self._crop_to_box(iso_img, box)

            # normalize (min-max → CLAHE)
            iso_img = self._normalize(iso_img)

            out = self.model.predict(image=iso_img[None,], voxel_spacing=(1.0,1.0,1.0))
            amap = out["anomaly_map"].squeeze()
            amap = 1.0 / (1.0 + np.exp(-np.clip(amap, -40, 40)))
        
            total_pathology = np.sum(amap[amap > class_threshold])
            print('sigmoid sum:', np.sum(amap), 'total_pathology', total_pathology)
        
            def prob(x):
                a = np.log(9) / (40000 - 1800)  # ≈ 5.75e-5
                return 1.0 / (1.0 + np.exp(-a * (np.asarray(x) - 1800)))
        
            if total_pathology < class_sum: #No pathology
                result['processing_status'] = 'Success' 
                result['pathology'] = 0
                result['probability_of_pathology'] = prob(total_pathology)
            
                return result, segment_nii, None, None, amap
        
            else:
                # threshold -> binary mask in iso space
                bin_mask = (amap > seg_threshold).astype(np.uint8)
                # boundary of connected clusters (1-voxel thick)
                boundary_mask = self._boundary_from_binary(bin_mask)

                # zoom factors from iso back to original
                factors_back = tuple(o / i for o, i in zip(orig_shape, boundary_mask.shape))
                mask_back = ndi_zoom(boundary_mask.astype(np.uint8), zoom=factors_back, order=0).astype(np.uint8)
                
                #Bbox of largest cluster
                _struct = generate_binary_structure(3, 3)
                lbl, ncomp = ndi_label(bin_mask, structure=_struct)
                if ncomp > 0:
                    counts = np.bincount(lbl.ravel())
                    counts[0] = 0  # background
                    largest_lab = int(counts.argmax())
                    largest_mask = (lbl == largest_lab).astype(np.uint8)
                else:
                    largest_mask = np.zeros_like(bin_mask, dtype=np.uint8)
            
                idxs = np.where(largest_mask)
                x_min_i, x_max_i = int(idxs[0].min()), int(idxs[0].max())
                y_min_i, y_max_i = int(idxs[1].min()), int(idxs[1].max())
                z_min_i, z_max_i = int(idxs[2].min()), int(idxs[2].max())
                bbox_iso = (x_min_i, x_max_i, y_min_i, y_max_i, z_min_i, z_max_i)
            
                if bbox_iso is not None:
                    # build the 8 corners of the iso-space bbox (use +1 on max to include the full voxel extent)
                    x0, x1, y0, y1, z0, z1 = bbox_iso
                    corners_iso = np.array([
                        [x0, y0, z0], [x1+1, y0, z0], [x0, y1+1, z0], [x0, y0, z1+1],
                        [x1+1, y1+1, z0], [x1+1, y0, z1+1], [x0, y1+1, z1+1], [x1+1, y1+1, z1+1]
                    ], dtype=np.float64)

                # scale iso indices back to original voxel index space
                scale = np.array(factors_back, dtype=np.float64)  # (sx, sy, sz)
                corners_orig_idx = corners_iso * scale  # still in voxel indices of the original array

                # per-axis min/max in patient space
                x_min_mm, y_min_mm, z_min_mm = corners_orig_idx.min(axis=0)
                x_max_mm, y_max_mm, z_max_mm = corners_orig_idx.max(axis=0)

                # requested output order: x_min,x_max,y_min,y_max,z_min,z_max
                bbox_mm = (float(x_min_mm), float(x_max_mm),
                   float(y_min_mm), float(y_max_mm),
                   float(z_min_mm), float(z_max_mm))
                
                
                #Find closest structure
                
                bin_mask_orig = ndi_zoom(bin_mask.astype(np.uint8), zoom=factors_back, order=0).astype(np.uint8)
                cnt = np.bincount(segmentation.ravel().astype(np.int64), weights=bin_mask_orig.ravel().astype(np.float32))  
                lbl = cnt[1:].argmax() + 1
                most_intersected_structure = self.total_dict[lbl]

                # Pack outputs as NIfTI in original geometry
                img_out = nib.Nifti1Image(image.astype(np.float32), orig_affine, header=nii.header)
                mask_out = nib.Nifti1Image(mask_back.astype(np.uint8), orig_affine, header=nii.header)
            
                result['processing_status'] = 'Success' 
                result['pathology'] = 1
                result['probability_of_pathology'] = prob(total_pathology)
                result['pathology_localization'] = bbox_mm
                result['most_dangerous_pathology_type'] = 'Anomaly of ' + most_intersected_structure


                return result, segment_nii, img_out, mask_out, amap
            
        except Exception as e:
            print(str(e))
            result['processing_status'] = 'Failure'
            result['debug_message'] = f'General error {str(e)}'
            return result, None, None, None, None
                
