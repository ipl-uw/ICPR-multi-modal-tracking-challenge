import torch
import numpy as np
import os
import os.path as osp
import pdb
import cv2
from sam2.build_sam import build_sam2_video_predictor
from tqdm import tqdm

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
data_path = "../../data"
out_path = "../../output/sam2.1"
vis_path = "../../vis/sam2.1_mask"
modal='color'
vis = True
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

os.makedirs(vis_path, exist_ok=True)

seqs = sorted(os.listdir(data_path))

for seq in (seqs):

    print(seq)

    # if os.path.exists(f'{out_path}/{seq}/Prediction.txt'):
    #     continue

    height, width = cv2.imread(f'{data_path}/{seq}/{modal}/00000001.png').shape[:2]

    if vis:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(osp.join(vis_path,f'{seq}_{modal}.mp4'), fourcc, 20, (width, height))

    os.makedirs(f'{out_path}/{seq}', exist_ok=True)

    x, y, w, h = np.loadtxt(f'{data_path}/{seq}/groundtruth.txt', delimiter=',').tolist()
    first_box = [x, y, x+w, y+h]
    results = [[x, y, w, h]]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(osp.join(data_path, seq, modal), offload_video_to_cpu=True, offload_state_to_cpu=True)
    
        frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, box=first_box, frame_idx=0, obj_id=1)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask = masks[0, 0].cpu().numpy()
            mask = mask > 0.0
            non_zero_indices = np.argwhere(mask)
            
            if len(non_zero_indices) == 0:
                bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                bbox = [x_min, y_min, x_max-x_min, y_max-y_min]

            if frame_idx != 0:
                results.append(bbox)

            if vis:
                img = cv2.imread(f'{data_path}/{seq}/{modal}/{frame_idx+1:08d}.png')
                 # Convert the binary mask to 3 channels
                mask_img = np.zeros_like(img)
                mask_img[mask] = [0, 255, 0]  # Green color for the mask
                
                # Optionally, blend the original image with the mask
                alpha = 0.5  # Transparency factor
                blended = cv2.addWeighted(img, 1 - alpha, mask_img, alpha, 0)

                # Draw the bounding box on the blended image
                cv2.rectangle(blended, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

                # Save the resulting frame
                out.write(blended)
    
    if vis:
        out.release()

    with open(f'{out_path}/{seq}/Prediction.txt', 'w') as f:
        for result in results:
            f.write(','.join([str(int(i)) for i in result]) + '\n')

    predictor.reset_state(state)
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()