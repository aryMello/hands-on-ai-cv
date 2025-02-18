import torch
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

def detect(frame, net, transform):
    height, width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2, 0, 1)
    x = x.unsqueeze(0)  # Add batch dimension
    x = x.float()  # Ensure tensor is of float type
    
    # Make sure the model is in evaluation mode and on the correct device
    net.eval()
    
    # Run the model forward pass
    with torch.no_grad():
        y = net(x)

    detections = y.data
    scale = torch.Tensor([width, height, width, height])

    # detections = [batch, number of classes, number of occurrences, (score, x0, y0, x1, y1)]
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= 0.6:  # Confidence threshold
            pt = (detections[0, i, j, 1:] * scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1

    return frame

# Creating the SSD neural network
net = build_ssd('test')
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth', map_location='cpu'))  # Adjust for your device if needed

# Creating the transformation
transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('epic_horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('output.mp4', fps=fps)

for i, frame in enumerate(reader):
    frame = detect(frame, net, transform)
    writer.append_data(frame)
    print(i)

writer.close()
