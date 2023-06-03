
import clip, os, torch, cv2, numpy as np
from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import * 


class SAMCLIPInstanceSegmentation(object): 
    def __init__(self):
        self.sam = None
        self.clip_model = None
        self.clip_preprocess = None
        self.cur_img_np = None
        self.overlay_img = None
        self.device = None
    
    def load_set_clip(self, clip_version="ViT-B/32"):
        """
        sets clip models that will be used to get text/image encodings
        
        clip_version (str): string of clip type to load
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

    def load_set_sam(self, sam_pth="model/sam_vit_h_4b8939.pth"):
        """
        sets SAM model which will segment the image
        
        sam_pth (str): path to SAM model
        """
        
        if os.path.exists(sam_pth) == False:
            raise Exception(f"Did not find sam file path: {sam_pth}")
        self.sam = build_sam(checkpoint=sam_pth)
        
    def cosine_dist_img_txt(self, cropped_boxes, class_prompt):
        """
        get list of cosine distances between all cropped boxes single class prompt
        
        cropped_boxes (list): list of boxes that SAM segmented, indices align with returnd list 
        class_prompt (str): class prompt txt
        
        return (list): list of cosine distances of each cropped box to prompt, indices align with cropped_boxes
        """
        cosine_distances = []
        text = clip.tokenize(class_prompt).to(self.device)
        for cropped_box_img in cropped_boxes:  # TODO don't loop through these images
            image = self.clip_preprocess(cropped_box_img).unsqueeze(0).to(self.device)
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)
            img_features = image_features.cpu().detach().numpy()
            txt_features = text_features.cpu().detach().numpy()

            cosine_dist = cosine_similarity(img_features, txt_features)[0][0]
            cosine_distances.append(cosine_dist)
        return cosine_distances
    
    def get_optimal_font_scale(self, text, width):
        """
        get optimal font scale for writing overlay
        
        text (str): string that will be displayed on overlay 
        width (int): width of box
        
        return (float): optimal font scale size
        """
        for scale in reversed(range(0, 60, 1)):
            textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
            new_width = textSize[0][0]
            if (new_width <= width):
                return scale/10
        return 1  # TODO move this function to utils

    def generate_image_sam_mask(self, img_pth, stability_score_threshold=.98, predicted_iou_threshold=.98):
        """
        Generate and return SAM masks
        
        img_pth (str): path to image to run detections
        stability_score_threshold (float): filtering threshold for stability/quality of SAM masks
        predicted_iou_threshold (float): model's (SAM's) own prediction of qualtiy
        
        
        return (list): list of SAM masks
        
        """
         
        img = Image.open(img_pth)
        self.cur_img_np = np.asarray(img)
        self.cur_img_pth = img_pth

        mask_generator = SamAutomaticMaskGenerator(self.sam, points_per_side=8)
        masks = mask_generator.generate(self.cur_img_np)

        og_masks = masks.copy()
        
        masks = []
        for j in range(0, len(og_masks)):
            if og_masks[j]["stability_score"] > stability_score_threshold and og_masks[j]["predicted_iou"] > predicted_iou_threshold:
                masks.append(og_masks[j])
                
        return masks
    
    def get_overlay_img(self, masks, classes, obj_threshold=.27):
        """
        get image numpy array overlayed with detections
        
        masks (list): list of SAM mask dictionaries
        classes (list): list of attempted detected classes
        
        return (numpy array): image numpy array with overlays
        """
        # TODO maybe set masks instead of pass in?
        # TODO break this into multiple functions
        FONT_SCALE = 2e-3
        
        # Cut out all masks
        cropped_boxes = []

        for mask in masks:
            cropped_boxes.append(segment_image(self.cur_img_np, mask["segmentation"]).crop(convert_box_xywh_to_xyxy(mask["bbox"])))

        class_prompts = []
        for cls in classes: 
            class_prompts.append(f"a photo of a {cls}")   
            
        img_og = cv2.imread(self.cur_img_pth)
            
        for idx, class_prompt in enumerate(class_prompts):

            scores = self.cosine_dist_img_txt(cropped_boxes, class_prompt)
            indices = get_indices_of_values_above_threshold(scores, obj_threshold)

            segmentation_masks = []
            polygons = []

            for seg_idx in indices:
                cntrs, _ = cv2.findContours(masks[seg_idx]["segmentation"].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                polygons.append(cntrs)
                segmentation_mask_image = Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255)
                segmentation_masks.append(segmentation_mask_image)




            height, width, _ = img_og.shape

            font_scale = min(width, height) * FONT_SCALE

            for j in range(len(polygons)):
                poly = polygons[j]
                img_og = cv2.polylines(img=img_og, pts=poly, isClosed=True, color=(0, 255, 0), thickness=3)

                label = f"{classes[idx]}__{round(float(scores[indices[j]]), 3)}"

                max_poly_idx = 0
                max_poly_area = -1
                for i in range(len(poly)):
                    area = cv2.contourArea(poly[i])
                    if area > max_poly_area:
                       max_poly_idx = i
                       max_poly_area = area

                M = cv2.moments(poly[max_poly_idx])
                if M['m00'] == 0:
                    center_full_x = 0
                    center_full_y = 0
                else:
                    center_full_x = M['m10'] / M['m00']
                    center_full_y = M['m01'] / M['m00']
                    
                width = poly[i][:,:,1].max() - poly[i][:,:,1].min()  # TODO check if hight or width 
            
                font_scale = self.get_optimal_font_scale(label, width)

                cv2.putText(
                    img_og,
                    label,
                    org=(int(center_full_x), int(center_full_y)),
                    fontFace=0,
                    fontScale=font_scale,
                    color=(255, 255, 255),
                    thickness=3
                )

        return img_og

