import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn.functional as F

# Initialize the polygon state
def initialize_polygon():
    """
    Initializes the polygon state.

    Returns:
        dict: A dictionary with 'points' and 'closed' status.
    """
    return {'points': [], 'closed': False}

# Add a point to the polygon when the user clicks on the image
def add_point(img_original, polygon_state, evt: gr.SelectData):
    """
    Adds a point to the polygon based on user click event.

    Args:
        img_original (PIL.Image): The original image.
        polygon_state (dict): The current state of the polygon.
        evt (gr.SelectData): The click event data.

    Returns:
        tuple: Updated image with polygon and updated polygon state.
    """
    if polygon_state['closed']:
        return img_original, polygon_state  # Do not add points if polygon is closed

    x, y = evt.index
    polygon_state['points'].append((x, y))

    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)

    # Draw lines between points
    if len(polygon_state['points']) > 1:
        draw.line(polygon_state['points'], fill='red', width=2)

    # Draw points
    for point in polygon_state['points']:
        draw.ellipse((point[0]-3, point[1]-3, point[0]+3, point[1]+3), fill='blue')

    return img_with_poly, polygon_state

# Close the polygon when the user clicks the "Close Polygon" button
def close_polygon(img_original, polygon_state):
    """
    Closes the polygon if there are at least three points.

    Args:
        img_original (PIL.Image): The original image.
        polygon_state (dict): The current state of the polygon.

    Returns:
        tuple: Updated image with closed polygon and updated polygon state.
    """
    if not polygon_state['closed'] and len(polygon_state['points']) > 2:
        polygon_state['closed'] = True
        img_with_poly = img_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        draw.polygon(polygon_state['points'], outline='red')
        return img_with_poly, polygon_state
    else:
        return img_original, polygon_state

# Update the background image by drawing the shifted polygon on it
def update_background(background_image_original, polygon_state, dx, dy):
    """
    Updates the background image by drawing the shifted polygon on it.

    Args:
        background_image_original (PIL.Image): The original background image.
        polygon_state (dict): The current state of the polygon.
        dx (int): Horizontal offset.
        dy (int): Vertical offset.

    Returns:
        PIL.Image: The updated background image with the polygon overlay.
    """
    if background_image_original is None:
        return None

    if polygon_state['closed']:
        img_with_poly = background_image_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        shifted_points = [(x + dx, y + dy) for x, y in polygon_state['points']]
        draw.polygon(shifted_points, outline='red')
        return img_with_poly
    else:
        return background_image_original

# Create a binary mask from polygon points
def create_mask_from_points(points, img_h, img_w):
    """
    Creates a binary mask from the given polygon points.

    Args:
        points (np.ndarray): Polygon points of shape (n, 2).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        np.ndarray: Binary mask of shape (img_h, img_w).
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(points) > 0:
        polygon = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        mask_image = Image.new('L', (img_w, img_h), 0)
        draw = ImageDraw.Draw(mask_image)
        draw.polygon([tuple(point) for point in points], outline=255, fill=255)
        mask = np.array(mask_image, dtype=np.uint8)

    return mask

# Calculate the Laplacian loss between the foreground and blended image
def cal_laplacian_loss(source_img, mask, blended_img, background_img):
    """
    Computes a masked Laplacian alignment loss for Poisson blending.

    Args:
        source_img (torch.Tensor): Source image tensor.
        mask (torch.Tensor): Foreground mask tensor.
        blended_img (torch.Tensor): Optimized blended image tensor.
        background_img (torch.Tensor): Background image tensor.

    Returns:
        torch.Tensor: The computed Laplacian loss.
    """
    target_h, target_w = blended_img.shape[-2:]
    if source_img.shape[-2:] != (target_h, target_w):
        source_img = F.interpolate(source_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
    if background_img.shape[-2:] != (target_h, target_w):
        background_img = F.interpolate(background_img, size=(target_h, target_w), mode='bilinear', align_corners=False)
    if mask.shape[-2:] != (target_h, target_w):
        mask = F.interpolate(mask, size=(target_h, target_w), mode='nearest')

    laplacian_kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=source_img.device,
        dtype=source_img.dtype,
    ).view(1, 1, 3, 3)

    channels = source_img.shape[1]
    laplacian_kernel = laplacian_kernel.repeat(channels, 1, 1, 1)

    source_laplacian = F.conv2d(source_img, laplacian_kernel, padding=1, groups=channels)
    blended_laplacian = F.conv2d(blended_img, laplacian_kernel, padding=1, groups=channels)

    mask = mask.expand_as(source_laplacian)
    boundary_mask = torch.clamp(F.max_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask, 0.0, 1.0)

    gradient_loss = torch.mean(torch.abs(source_laplacian - blended_laplacian) * mask)
    boundary_loss = torch.mean(torch.abs(blended_img - background_img) * boundary_mask)
    inside_loss = torch.mean(torch.abs(blended_img - source_img) * mask)

    return gradient_loss + 0.25 * boundary_loss + 0.01 * inside_loss

# Perform Poisson image blending
def blending(foreground_image_original, background_image_original, dx, dy, polygon_state):
    """
    Blends the foreground polygon area onto the background image using Poisson blending.

    Args:
        foreground_image_original (PIL.Image): The original foreground image.
        background_image_original (PIL.Image): The original background image.
        dx (int): Horizontal offset.
        dy (int): Vertical offset.
        polygon_state (dict): The current state of the polygon.

    Returns:
        np.ndarray: The blended image as a numpy array.
    """
    if not polygon_state['closed'] or background_image_original is None or foreground_image_original is None:
        return background_image_original  # Return original background if conditions are not met

    foreground_np = np.array(foreground_image_original)
    background_np = np.array(background_image_original)

    foreground_points = np.array(polygon_state['points'], dtype=np.int32)
    background_points = foreground_points + np.array([int(dx), int(dy)], dtype=np.int32)

    foreground_mask = create_mask_from_points(foreground_points, foreground_np.shape[0], foreground_np.shape[1])
    background_mask = create_mask_from_points(background_points, background_np.shape[0], background_np.shape[1])

    ys, xs = np.where(background_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return background_np

    margin = 18
    x_min = max(int(xs.min()) - margin, 0)
    y_min = max(int(ys.min()) - margin, 0)
    x_max = min(int(xs.max()) + margin + 1, background_np.shape[1])
    y_max = min(int(ys.max()) + margin + 1, background_np.shape[0])

    source_canvas = np.zeros_like(background_np)
    fg_h, fg_w = foreground_np.shape[:2]
    bg_h, bg_w = background_np.shape[:2]
    x1 = max(0, int(dx))
    y1 = max(0, int(dy))
    x2 = min(bg_w, int(dx) + fg_w)
    y2 = min(bg_h, int(dy) + fg_h)
    sx1 = max(0, -int(dx))
    sy1 = max(0, -int(dy))
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    if x1 < x2 and y1 < y2:
        source_canvas[y1:y2, x1:x2] = foreground_np[sy1:sy2, sx1:sx2]

    source_crop = source_canvas[y_min:y_max, x_min:x_max].copy()
    target_crop = background_np[y_min:y_max, x_min:x_max].copy()
    mask_crop = background_mask[y_min:y_max, x_min:x_max].copy()

    if mask_crop.sum() == 0:
        return background_np

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    source_tensor = torch.from_numpy(source_crop).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    target_tensor = torch.from_numpy(target_crop).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    mask_tensor = torch.from_numpy(mask_crop).to(device).unsqueeze(0).unsqueeze(0).float() / 255.

    neighbor_kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, 0.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=device,
        dtype=source_tensor.dtype,
    ).view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    source_guidance = 4.0 * source_tensor - F.conv2d(source_tensor, neighbor_kernel, padding=1, groups=3)
    source_guidance = source_guidance * mask_tensor

    blended_tensor = target_tensor.clone()
    iter_num = 180
    for step in range(iter_num):
        neighbor_sum = F.conv2d(blended_tensor, neighbor_kernel, padding=1, groups=3)
        updated = (neighbor_sum + source_guidance) / 4.0
        blended_tensor = updated * mask_tensor + target_tensor * (1.0 - mask_tensor)

        if step % 30 == 0:
            loss = cal_laplacian_loss(source_tensor, mask_tensor, blended_tensor, target_tensor)
            print(f'Jacobi step: {step}, Laplacian distance loss: {loss.item()}')

    result = background_np.copy()
    blended_crop = torch.clamp(blended_tensor.detach(), 0, 1).cpu().permute(0, 2, 3, 1).squeeze().numpy() * 255
    result[y_min:y_max, x_min:x_max] = blended_crop.astype(np.uint8)
    return result

# Helper function to close the polygon and reset dx
def close_polygon_and_reset_dx(img_original, polygon_state, dx, dy, background_image_original):
    """
    Closes the polygon, resets dx to 0, and updates the background image.

    Args:
        img_original (PIL.Image): The original image.
        polygon_state (dict): The current state of the polygon.
        dx (int): Horizontal offset.
        dy (int): Vertical offset.
        background_image_original (PIL.Image): The original background image.

    Returns:
        tuple: Updated image with polygon, updated polygon state, updated background image, and reset dx value.
    """
    # Close polygon
    img_with_poly, updated_polygon_state = close_polygon(img_original, polygon_state)

    # Reset dx value to 0
    new_dx = gr.update(value=0)

    # Update background image
    updated_background = update_background(background_image_original, updated_polygon_state, 0, dy)
    return img_with_poly, updated_polygon_state, updated_background, new_dx

# Gradio Interface
with gr.Blocks(title="Poisson Image Blending", css="""
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .gr-button {
        font-size: 1em;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        background-color: #6200ee;
        color: #ffffff;
        border: none;
    }
    .gr-button:hover {
        background-color: #3700b3;
    }
    .gr-slider input[type=range] {
        accent-color: #03dac6;
    }
    .gr-text, .gr-markdown {
        font-size: 1.1em;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #bb86fc;
    }
    .gr-input, .gr-output {
        background-color: #2c2c2c;
        border: 1px solid #3c3c3c;
    }
""") as demo:
    # Initialize states
    polygon_state = gr.State(initialize_polygon())
    background_image_original = gr.State(value=None)

    # Title and description
    gr.Markdown("<h1 style='text-align: center;'>Poisson Image Blending</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 1.2em;'>Blend a selected area from a foreground image onto a background image with adjustable positions.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Foreground Image")
            foreground_image_original = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the foreground image where the polygon will be selected.</p>"
            )
            gr.Markdown("### Foreground Image with Polygon")
            foreground_image_with_polygon = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Click on the image to define the polygon area. After selecting at least three points, click <strong>Close Polygon</strong>.</p>"
            )
            close_polygon_button = gr.Button("Close Polygon")
        with gr.Column():
            gr.Markdown("### Background Image")
            background_image = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown("<p style='font-size: 0.9em;'>Upload the background image where the polygon will be placed.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Background Image with Polygon Overlay")
            background_image_with_polygon = gr.Image(
                label="", type="pil", height=500
            )
            gr.Markdown("<p style='font-size: 0.9em;'>Adjust the position of the polygon using the sliders below.</p>")
        with gr.Column():
            gr.Markdown("### Blended Image")
            output_image = gr.Image(
                label="", type="pil", height=500  # Increased height for larger display
            )

    with gr.Row():
        with gr.Column():
            dx = gr.Slider(
                label="Horizontal Offset", minimum=-500, maximum=500, step=1, value=0
            )
        with gr.Column():
            dy = gr.Slider(
                label="Vertical Offset", minimum=-500, maximum=500, step=1, value=0
            )
        blend_button = gr.Button("Blend Images")

    # Interactions

    # Copy the original image to the interactive image when uploaded
    foreground_image_original.change(
        fn=lambda img: img,
        inputs=foreground_image_original,
        outputs=foreground_image_with_polygon,
    )

    # User interacts with the image with polygon
    foreground_image_with_polygon.select(
        add_point,
        inputs=[foreground_image_original, polygon_state],
        outputs=[foreground_image_with_polygon, polygon_state],
    )

    close_polygon_button.click(
        fn=close_polygon_and_reset_dx,
        inputs=[foreground_image_original, polygon_state, dx, dy, background_image_original],
        outputs=[foreground_image_with_polygon, polygon_state, background_image_with_polygon, dx],
    )

    background_image.change(
        fn=lambda img: img,
        inputs=background_image,
        outputs=background_image_original,
    )

    # Update background image when dx or dy changes
    dx.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )
    dy.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )

    # Blend images when button is clicked
    blend_button.click(
        fn=blending,
        inputs=[foreground_image_original, background_image_original, dx, dy, polygon_state],
        outputs=output_image,
    )

# Launch the Gradio app
demo.launch()
