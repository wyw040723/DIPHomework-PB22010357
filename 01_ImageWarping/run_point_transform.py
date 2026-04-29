import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image


def _mls_affine_inverse_map(grid_xy, source_pts, target_pts, alpha=1.0, eps=1e-8):
    source_pts = np.asarray(source_pts, dtype=np.float32)
    target_pts = np.asarray(target_pts, dtype=np.float32)

    if source_pts.shape[0] == 0 or target_pts.shape[0] == 0:
        return grid_xy.astype(np.float32)

    count = min(source_pts.shape[0], target_pts.shape[0])
    source_pts = source_pts[:count]
    target_pts = target_pts[:count]

    diffs = grid_xy[:, None, :] - target_pts[None, :, :]
    dist_sq = np.sum(diffs * diffs, axis=2)

    exact_mask = np.min(dist_sq, axis=1) < 1e-6
    mapped_xy = np.empty_like(grid_xy, dtype=np.float32)

    if np.any(~exact_mask):
        active_xy = grid_xy[~exact_mask]
        active_diff = active_xy[:, None, :] - target_pts[None, :, :]
        active_dist_sq = np.sum(active_diff * active_diff, axis=2)

        weights = 1.0 / (np.power(active_dist_sq, alpha) + eps)
        weight_sum = np.sum(weights, axis=1, keepdims=True)

        target_centroid = np.sum(weights[:, :, None] * target_pts[None, :, :], axis=1) / weight_sum
        source_centroid = np.sum(weights[:, :, None] * source_pts[None, :, :], axis=1) / weight_sum

        target_centered = target_pts[None, :, :] - target_centroid[:, None, :]
        source_centered = source_pts[None, :, :] - source_centroid[:, None, :]

        a11 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 0], target_centered[:, :, 0])
        a12 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 0], target_centered[:, :, 1])
        a21 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 1], target_centered[:, :, 0])
        a22 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 1], target_centered[:, :, 1])

        b11 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 0], source_centered[:, :, 0])
        b12 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 0], source_centered[:, :, 1])
        b21 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 1], source_centered[:, :, 0])
        b22 = np.einsum("nk,nk,nk->n", weights, target_centered[:, :, 1], source_centered[:, :, 1])

        det = a11 * a22 - a12 * a21
        det = np.where(np.abs(det) < eps, eps, det)

        m11 = (a22 * b11 - a12 * b21) / det
        m12 = (a22 * b12 - a12 * b22) / det
        m21 = (-a21 * b11 + a11 * b21) / det
        m22 = (-a21 * b12 + a11 * b22) / det

        offset = active_xy - target_centroid
        mapped_xy[~exact_mask, 0] = offset[:, 0] * m11 + offset[:, 1] * m21 + source_centroid[:, 0]
        mapped_xy[~exact_mask, 1] = offset[:, 0] * m12 + offset[:, 1] * m22 + source_centroid[:, 1]

    if np.any(exact_mask):
        exact_target = grid_xy[exact_mask]
        exact_diff = exact_target[:, None, :] - target_pts[None, :, :]
        exact_idx = np.argmin(np.sum(exact_diff * exact_diff, axis=2), axis=1)
        mapped_xy[exact_mask] = source_pts[exact_idx]

    return mapped_xy.astype(np.float32)

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    if image is None:
        return None

    image = np.asarray(image)

    if source_pts is None or target_pts is None:
        return image

    if len(source_pts) == 0 or len(target_pts) == 0:
        return image

    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32),
                                 np.arange(height, dtype=np.float32))
    grid_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)

    source_map = _mls_affine_inverse_map(grid_xy,
                                         source_pts,
                                         target_pts,
                                         alpha=alpha,
                                         eps=eps)

    map_x = source_map[:, 0].reshape(height, width).astype(np.float32)
    map_y = source_map[:, 1].reshape(height, width).astype(np.float32)

    if image.ndim == 2:
        border_value = 255
    else:
        border_value = tuple([255] * image.shape[2])

    return cv2.remap(image,
                     map_x,
                     map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT,
                     borderValue=border_value)

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()