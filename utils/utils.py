import numpy as np, clip
from PIL import Image


def convert_box_xywh_to_xyxy(box):
    """
    convert bounding box from SAM format to x1,y1,x2,y2
    
    box (list) : list in SAM bounding box format of x1,y1,width,hight
    
    return (list): list in bounding box format x1,y1,x2,y2
    """
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]

def segment_image(img_np, segmentation_mask):
    """
    returns segmented image 
    
    img_np (numpy array): numpy array of entire image
    segmentation_mask: numpy array of binarys for segmented mask
    
    """
    segmented_img_np = np.zeros_like(img_np)
    segmented_img_np[segmentation_mask] = img_np[segmentation_mask]
    segmented_image = Image.fromarray(segmented_img_np)  # pixels black but segment
    return segmented_image



def get_indices_of_values_above_threshold(values, threshold):
    """
    get indicies of bounding box crops that are above cosine distance threshold for prompt
    
    values (list): list of cosine distances betweeen cropped boxes and text
    threshold (float): cosine distance threshold to count as class
    """
    return [i for i, v in enumerate(values) if v > threshold]


# def cosine_dist_img_txt(cropped_boxes, single_class):
#     cosine_distances = []
#     text = clip.tokenize(single_class).to(device)
#     for cropped_box_img in cropped_boxes:  # TODO don't loop through these images
#         image = preprocess(cropped_box_img).unsqueeze(0).to(device)
#         image_features = model.encode_image(image)
#         text_features = model.encode_text(text)
#         img_features = image_features.cpu().detach().numpy()
#         txt_features = text_features.cpu().detach().numpy()
        
#         cosine_dist = cosine_similarity(img_features, txt_features)[0][0]
#         cosine_distances.append(cosine_dist)
#     return cosine_distances

# def crop_image(img_np, mask):
#     x1,y1,x2,y2 = convert_box_xywh_to_xyxy(mask["bbox"])
#     cropped_img = Image.fromarray(img_np[y1:y2, x1:x2])