import mayavi.mlab as mlab
import cv2
import numpy as np
def draw_gt_boxes3d(gt_boxes3d, fig, color=(1, 1, 1)):
    """
    Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (3,8) for XYZs of the box corners
        fig: figure handler
        color: RGB value tuple in range (0,1), box line color
    """
    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)

        i, j = k, k + 4
        mlab.plot3d([gt_boxes3d[0, i], gt_boxes3d[0, j]], [gt_boxes3d[1, i], gt_boxes3d[1, j]],
                    [gt_boxes3d[2, i], gt_boxes3d[2, j]], tube_radius=None, line_width=2, color=color, figure=fig)
    return fig