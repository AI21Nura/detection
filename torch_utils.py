import torchvision
import torch
import cv2
import os


COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
model.eval().to(device)

def get_filename(img_path):
  path, filename = os.path.split(img_path)
  fname, extension = os.path.splitext(filename)
  new_filename = fname + "_detect" + extension 
  new_img_path = os.path.join(path, new_filename)
  return new_filename, new_img_path


def draw_bbox(img_numpy, preds, img_path, conf_thresh = 0.8):
  bboxes = preds['boxes'][preds['scores'] > conf_thresh].detach().numpy()
  labels = preds['labels'][preds['scores'] > conf_thresh].detach()
  thcknss = 2 if (img_numpy.shape[0] * img_numpy.shape[1]) < 5e5 else 3

  for box, label in zip(bboxes, labels):
    class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
    (w, h), _ = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, img_numpy.shape[0]/1072*1.8, 3)
    pt1 = tuple([int(round(box[0]) ), int(round( box[1] )) ])
    pt2 = tuple([int(round(box[2]) ), int(round( box[3] )) ])

    if box[1] < h:
      pt3 = tuple([int(round(box[0]) ), int(round( box[1])) ])   
      pt4 = tuple([int(round(box[0]+w) ), int(round( box[1] + h)) ]) 
      pt5 = tuple([int(round(box[0]) ), int(round( box[1] + h)) ]) 
    else:
      pt3 = tuple([int(round(box[0]) ), int(round( box[1] - h)) ])   
      pt4 = tuple([int(round(box[0]+w) ), int(round( box[1])) ]) 
      pt5 = pt1

    cv2.rectangle(img_numpy, pt1, pt2,color=(255,0,0), thickness=3)
    cv2.rectangle(img_numpy, pt3, pt4,color=(255,0,0), thickness=-1)
    cv2.putText(img=img_numpy, text=class_name, org=pt5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=img_numpy.shape[0]/1072*1.8, color=(255,255,255), thickness=thcknss)
    
  new_filename, new_img_path = get_filename(img_path)
  cv2.imwrite(new_img_path, img_numpy[:,:,::-1])
  
  return new_filename


def get_predictions(img_numpy):
  img = torch.from_numpy(img_numpy.astype('float32')).permute(2,0,1).to(device) / 255
  predictions = model(img[None,...])
  return predictions[0]

def predict(img_path):
    img_numpy = cv2.imread(img_path)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_BGR2RGB)
    preds = get_predictions(img_numpy)
    return draw_bbox(img_numpy, preds, img_path)

