from matplotlib import pyplot as plt
import torch

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
            [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box coordinates from (center_x, center_y, width, height) format to (x_min, y_min, x_max, y_max) format.

    Args:
        x (torch.Tensor): Bounding box coordinates in (center_x, center_y, width, height) format.

    Returns:
        torch.Tensor: Bounding box coordinates in (x_min, y_min, x_max, y_max) format.
    """
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    """
    Rescales the bounding boxes based on the given output bounding box and image size.

    Args:
        out_bbox (torch.Tensor): The output bounding box.
        size (tuple): The size of the image in the format (width, height).

    Returns:
        torch.Tensor: The rescaled bounding boxes.
    """
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def plot_results(pil_img, prob, boxes, id2label):
    """
    Plot the results of object detection on an image.

    Args:
        pil_img (PIL.Image.Image): The input image.
        prob (torch.Tensor): The predicted probabilities of each object class.
        boxes (torch.Tensor): The predicted bounding boxes of each object.
        id2label (dict): A dictionary mapping class IDs to labels.

    Returns:
        None
    """
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

def visualize_predictions(image, outputs, id2label, threshold=0.9, keep_highest_scoring_bbox=False):
    """
    Visualizes the predictions on an image.

    Args:
        image (PIL.Image.Image): The input image.
        outputs (torch.Tensor): The model's output tensor.
        id2label (dict): A dictionary mapping class IDs to labels.
        threshold (float, optional): The confidence threshold for keeping predictions. Defaults to 0.9.
        keep_highest_scoring_bbox (bool, optional): Whether to keep only the highest scoring bounding box. Defaults to False.
    """
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold 
    if keep_highest_scoring_bbox:
        keep = probas.max(-1).values.argmax()
        keep = torch.tensor([keep])

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    # plot results
    plot_results(image, probas[keep], bboxes_scaled, id2label)