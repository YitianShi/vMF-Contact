import os
import numpy as np
import cv2
import glob
import open3d as o3d
from lang_sam import LangSAM
import time
from PIL import Image

langsam = LangSAM()

# Function to mark duplicates with a number
def mark_duplicates(labels):
    label_count = {}
    result = []
    
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
        result.append(f"{label}_{label_count[label]}")
    
    return result

def generate_langsam(pcd, img, target_object):
    # Process the prompt point cloud
    time_curr = time.time()

    masked_pcd_dict = {}
    # predict masks with lang_sam
    results = langsam.predict([Image.fromarray(img)], [". ".join(target_object)])

    print(f"Time taken for inference: {time.time() - time_curr}")
    
    # check if there are labels detected
    labels = results[0]["labels"]
    if len(labels) == 0:
        print("No labels detected.")
        return pcd
    # check duplicates in labels, if there are duplicates, mark them with a number
    labels = mark_duplicates(labels)
    print("Results: ", labels)
    print("Scores: ", results[0]["scores"])

    # mask point cloud and image
    for i, text in enumerate(labels):
        mask = results[0]["masks"][i].astype(np.uint8)[:, :, None]
        # mask image and point cloud
        pcd_masked = pcd[mask.reshape(-1) == 1]
        masked_pcd_dict[text] = pcd_masked
        # save masked image
        # cv2.imwrite(f"{current_file_folder}/{text}.jpg", img[..., ::-1] * mask)
    pcd_from_prompt = []
    for label in masked_pcd_dict:
        pcd_from_prompt.append(masked_pcd_dict[label])
        break
    pcd_from_prompt = np.concatenate(pcd_from_prompt, axis=0)
    # pcd_from_prompt = (pcd_from_prompt - pcd_shift)
    print("Prompt point cloud: ", pcd_from_prompt.shape)
    return pcd_from_prompt


def read_data(data_dir="/home/yitian/Research/dataset/data_active_grasp/"):
    # Read data from data_dir
    # data_dir: the directory of the data
    # return: a list of data
    data_path = glob.glob(data_dir + '*.npz')
    names = [file.split('/')[-1].split('.npz')[0] for file in data_path]
    position = np.array([[name.split("_")[i] for i in range(3)] for name in names], dtype=np.float32)
    for file in glob.glob(data_dir + '*.npz'):
        data = np.load(file)
        rgb = data['image']
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = data['depth']
        depth = np.uint8(depth / depth.max() * 255)

        pcd = data['pcd']
        pcd = pcd[(pcd[:, 2] > 0.035) & (pcd[:, 2] < 0.45)]
                
        # generate lang_sam
        target_object = ""
        while target_object == "":
            target_object = input("Please enter what you would like to grasp: ")
            target_object = target_object.split(".")
            print("Prompt: ", target_object)
        pcd_from_prompt = generate_langsam(pcd, rgb, target_object)
        pcd_vis = o3d.geometry.PointCloud()
        pcd_vis.points = o3d.utility.Vector3dVector(pcd_from_prompt)
        o3d.visualization.draw_geometries([pcd_vis])
        # show the image
        cv2.imshow('rgb', rgb)
        cv2.imshow('depth', depth)
        cv2.waitKey(0)

if __name__ == '__main__':
    read_data()