"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face 

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""

import numpy as np
import torch
from FLAME import FLAME
import pyrender
import trimesh
from config import get_config
import os
import pymeshlab
from renderer import *
import cv2
from tensor_cropper import transform_points

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image*255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1,2,0)[:,:,[2,1,0]]
    return image.astype(np.uint8).copy()

def main():
    config = get_config()
    set_rasterizer('pytorch3d')
    device = 'cuda'
    radian = np.pi/180.0
    config.batch_size = 1
    flamelayer = FLAME(config)
    h = 1024
    w = 1024
    # Creating a batch of mean shapes
    shape_params = torch.zeros(1,100).cuda()

    # Creating a batch of different global poses
    # pose_params_numpy[:, :3] : global rotaation yaw
    # pose_params_numpy[:, 3:] : jaw rotaation
    pose_params_numpy = np.array([[0.0*radian, 0.0*radian, 0.0*radian, 0.0, 0.0, 0.0]], dtype=np.float32)
    pose_params = torch.tensor(pose_params_numpy, dtype=torch.float32).cuda()

    # Cerating a batch of neutral expressions
    expression_params = torch.zeros(1, 50, dtype=torch.float32).cuda()
    flamelayer.cuda()
    
    # Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
    vertice, landmark = flamelayer(shape_params, expression_params, pose_params) # For RingNet project
    print(vertice.size(), landmark.size())

    # if config.optimize_eyeballpose and config.optimize_neckpose:
    #     neck_pose = torch.zeros(1, 3).cuda()
    #     eye_pose = torch.zeros(1, 6).cuda()
    #     vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose)

    neck_pose = torch.zeros(1, 3).cuda()
    eye_pose = torch.zeros(1, 6).cuda()
    render = SRenderY([h,w], obj_filename='head_template.obj', uv_size=256, rasterizer_type='pytorch3d').to(device)

    # Visualize Landmarks
    # This visualises the static landmarks and the pose dependent dynamic landmarks used for RingNet project
    faces = flamelayer.faces
    for j in range(50):
        print(j)
        for k in range(-10, 10):
            expression_params[0][j] = k
            vertice, landmark = flamelayer(shape_params, expression_params, pose_params, neck_pose, eye_pose) # For RingNet project
            vertices = vertice.detach().cpu().numpy().squeeze()
            vertices = torch.tensor([vertices], dtype=torch.float32, device=device)
            
            h = 1024
            w = 1024
            background = None
            cam = torch.tensor([[ 3.9265e+00, 0.003,  -0.003]], dtype=torch.float32, device=device)
            # cam = torch.tensor([[ 3.9265e+00, 1.6956e-03,  1.7192e-02]], dtype=torch.float32, device=device)
            # cam = codedict['cam']
            tform = None
            points_scale = [h, w]
            trans_verts = util.batch_orth_proj(vertices, cam); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
            # trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])

            # shape_images, _, grid, alpha_images = self.render.render_shape(vertices, trans_verts, h=h, w=w, images=background, return_grid=True)
            shape_images, _, grid, alpha_images = render.render_shape(vertices, trans_verts, h=h, w=w, images=background, return_grid=True)
            image = tensor2image(shape_images[0])
            cv2.putText(image, f'Expression{j}: {k}', (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), lineType=cv2.LINE_AA)
            cv2.imshow('mesh', image)
            key = cv2.waitKey(1) % 2 ** 16
            if key == ord('q') or key == ord('Q'):
                print('\'Q\' pressed, we are done here.')
                exit()
        expression_params[0][j] = 0
           


if __name__ == '__main__':
    main()