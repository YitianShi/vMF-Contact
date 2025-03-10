import numpy as np
from openai import OpenAI
import glob
import os
import json
import numpy as np
import time
from PIL import Image

current_file_folder = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
  import sys
  sys.path.append(current_file_folder)
  from img_bbox_utils import *
else:
  from .img_bbox_utils import *

######################################################################################################################  

def compile_expert_data(filename):
    data = np.load(filename, allow_pickle=True)
    base_filename = os.path.basename(filename)  # '-0.0219666975931492_30.npz'
    name_without_ext = os.path.splitext(base_filename)[0]  # '-0.0219666975931492_30'
    rot, elev = name_without_ext.split('_')  # '-0.0219666975931492', '30'

    image = data['image']
    image = crop_max_and_rotate(image, -float(rot))
    
    img_base64 = preprocess_image(image)

    bbox_center_dict = data['bbox_center_dict'].item()
    bbox_corners_dict = data['bbox_corners_dict'].item()
    bbox_lwh_dict = data['bbox_lwh_dict'].item()

    formatted_bbox_data = format_bbox_data(bbox_center_dict, bbox_corners_dict, bbox_lwh_dict)
    return img_base64, formatted_bbox_data, rot, elev

def json_scene_data(bbox_data, rot, elev):
  json_data = json.dumps(
      {
            "bbox data to robot base": bbox_data,
            "camera horizontal coordinates to scene object center":
            { 
              "elevation_angle": round(float(elev), 3),
              "rotation_angle": round(float(rot), 3)
            }
      }
  )
  return json_data

base64_image_exp = []
json_data_exp = []
target_occlusion_pairs_exp = {
    0:["yellow tennisball",["red cup", "orange cordeless drill"]], 
    1:["purple plum",["red cheezit box", "white bottle"]],
    # 2:["golfball",["yellow cup"]], 
    # 3:["white tennis ball",["white dominos box", "red cheezit box"]],
    # 4: ["green cube",["red cup", "white bottle"]],
}
for ind in target_occlusion_pairs_exp.keys():
  folder = f'{current_file_folder}/context/{ind}'
  files = glob.glob(f'{folder}/*.npz')
  # print(f"Target: {target_list[ind]}")
  for filename in files:
    img_base64_exp, formatted_bbox_data_exp, rot_exp, elev_exp = compile_expert_data(filename)

    base64_image_exp.append(img_base64_exp)
    json_data_exp.append(json_scene_data(formatted_bbox_data_exp, rot_exp, elev_exp))
    # plt.imshow(image)
    # plt.show()
    #print(f"Processed {name_without_ext}: {json_data_exp[-1]}")
del ind  

######################################################################################################################

def return_prompt_scene(img, target_object):
    base64_image = preprocess_image(img)

    return [
        # {
        # "role": "system",
        # "content": "Please analyze the provided image and generate a comprehensive list of all objects present within 0.5 meters, ignoring any noisy background elements. Use your internal chain-of-thought reasoning and perform a self-check to ensure every object is accurately identified, including those that might be largely occluded by other objects. Assume the roles of three experts—a scene analyst, a spatial relationship expert, and an object recognition specialist—each independently evaluating the image. Have these expert perspectives compare their outputs, merge any discrepancies, and decide on the most accurate descriptions. If you have access to a previously generated object list from another view, compare it with your current findings. For objects that appear in both lists but have different descriptions, merge their descriptors into a unified entry; if differences are significant, select the description that appears most accurate. Additionally, if you identify any new objects that were not present in the previous list, add them to the final list. Before outputting the final result, perform a thorough self-validation to ensure that all objects within the specified 0.5-meter region have been accounted for and that the output strictly adheres to the required JSON format. Do not include any additional text, explanations, or your internal chain-of-thought details in the final output, as this list will be used to generate instance masks with the SAM model."
        # },
        {
        "role": "user",
        "content": [
            #{"image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
            {"type": "image_url", 
             "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text",
             "text": """
              Analyze the provided image and generate a **comprehensive list of all objects within 0.5 meters**, ignoring any noisy background elements. This analysis will be performed through the independent evaluations of **three expert perspectives**:

              1. **Scene Analyst** - Identifies objects based on spatial context, occlusions, and visibility.
              2. **Spatial Relationship Expert** - Evaluates object depth, relative positioning, and interactions.
              3. **Object Recognition Specialist** - Classifies objects based on shape, texture, and known visual characteristics.

              ### **Processing Steps**
              1. **Independent Expert Evaluations**  
                - Each expert **independently** analyzes the scene and produces a preliminary list of detected objects, along with descriptive attributes.
                - Attributes should focus on **size, material, color, and spatial positioning**.

              2. **Comparison and Merging of Object Lists**  
                - Objects that **appear in both the current view and a previously generated object list** are **merged**, consolidating differing descriptors into a unified entry.
                - If discrepancies arise, the **most accurate** descriptor is selected based on detailed analysis.
                - **Newly detected objects** (not in prior lists) are added to the final set.
                - **Duplicate objects** (e.g., multiple cups) are **differentiated** based on **unique spatial relationships**.

              3. **Validation and JSON Formatting**
                - Perform a **thorough validation** to confirm that all objects **within 0.5 meters** are accounted for.
                - Ensure strict **JSON format adherence**:
                  - Each object is represented as a **JSON object** with exactly **two keys**:
                    - `"object"`: **Main identifier** (e.g., `"cup"`, `"block"`), **no more than 2 words**.
                    - `"descriptors"`: **An array containing exactly three adjectives or descriptive phrases** (e.g., `["red", "plastic", "near box"]`).
                - **No extra text, explanations, or chain-of-thought reasoning** should appear in the final output. \n
               
              """ +  
              f"""
              4. **Important**
                - **Please put a special focus on the object: {target_object}**. If it is detected, ensure to put it as the first dictionary.  
                - **Background object such as table should not be included** 
                \n
              """ 
              + 
              """
              ### **Final Output Example**
              ```json
              [
                {
                  "object": "cup",
                  "descriptors": ["red", "plastic", "near wooden block"]
                },
                {
                  "object": "tennis ball",
                  "descriptors": ["yellow", "partially visible", "inside red cup"]
                },
                {
                  "object": "wooden block",
                  "descriptors": ["rectangular", "light brown", "supporting wooden board"]
                },
                {
                  "object": "wooden board",
                  "descriptors": ["flat", "medium brown", "resting on two objects"]
                },
                {
                  "object": "cheese box",
                  "descriptors": ["red", "cardboard", "next to wooden block"]
                }
              ]
              """
              }]
          }
        ]

def return_prompt_ordered_grasp(img, target_object, json_data):
    base64_image = preprocess_image(img)
    return [
        {
        "role": "system",
        "content": """You are an advanced Vision-Language Model (VLM) assisting in active vision for robotic grasping. Your task is to determine the best sequence of objects to remove in order to maximize the visibility and graspability of the target object. The grasping network's confidence score remains below the threshold after three iterations of Next-Best-View (NBV) searching due to occlusions.  
              ### **Reasoning Process (Chain of Thought)**
              1. **Analyze the scene:** Identify objects that occlude the target object based on their relative positions and bounding box centers.  
              2. **Assess occlusion impact:** Determine which occluding object(s) contribute the most to the obstruction.  
              ### **Expected Output Format**  
              The output must be a JSON array of object labels in the correct removal sequence:  
              ```
              ["object_1", "object_2", ..., "object_n"]
              ```
              """
        },
        {
        "role": "user",
        "content": [
            #{"image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
            {"type": "image_url", 
             "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            {"type": "text",
             "text": f"""
              The attached image is the current view, the target object is the **{target_object}**, and the following json including the detected objects and their 3D position data: \n{json_data}\n
              
              The output must be a JSON array of object labels in the correct removal sequence without any extra text, explanations, or chain-of-thought reasoning.:  
              ```
              ["object_1", "object_2", ..., "object_n"]
              ```
              """ 
        }
        ]
        }
        ]

def return_prompt_guess(imgs, target_object:str, json_data):
    base64_images = [img if isinstance(img, str) else preprocess_image(img) for img in imgs]

    msgs = [{
        "role": "system",
        "content":
            """
              You are an advanced Vision-Language Model (VLM). You have been provided with two views of current scence, a list of objects detected in the scene by GroundingDINO. 
              Each object is described by its label name, its center coordinates (x,y,z), its Bbox size(length, width, height), and its 3D Bbox corner coordinate (eight vertex coordinates). All units are meter.
              Your goal is to determine which visible object is most likely blocking, hiding or totally occluding a currently invisible: """ + target_object + """ in the scene (including situations such as being covered or enclosed). To do so, follow these steps:
              Adopt the “Expert Mode” and pretend you are three different experts, each analyzing the scenario and potentially reaching different conclusions.
              Focus on relevant physical properties—such as shape, size, position, and coverage—rather than color-based similarities when deciding which object might be blocking, hiding or totally occluding the invisible target.
              Each of the three experts may arrive at their own candidate. Combine or vote on their conclusions, unifying the final response into one label.
              Use a hidden chain-of-thought to perform the internal reasoning without any extra text or explanations. 
              You have to at least provide one label name of the object that is most likely blocking, hiding or totally occluding the target object. **Important**: Please generate the list in descending order of your confidence.""" 
        }]
    
    # examples
    for i, (target_object_exp, occluding_obj_exp) in enumerate(target_occlusion_pairs_exp.values()):
    
      msgs.append({
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": f"Find the {target_object_exp} with following scene information: \n{json_data_exp[i*2+1]}\n```"
                  },
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/png;base64,{base64_image_exp[i*2]}",
                      }
                  },
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/png;base64,{base64_image_exp[i*2+1]}",
                      }
                  }
              ]
          })
      msgs.append(
              {
              "role": "assistant",
              "content": str(occluding_obj_exp) # + f" Reason: because they are possible to hide my target object: {target_occlusion_pairs_exp[i][0]}."
              }
              )
        
    # user input
    msgs.append(
        {
              "role": "user",
              "content": [
                  {
                      "type": "text",
                      "text": f"Find the {target_object.lower()} with following scene information: \n{json_data}\n```"
                  },
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/png;base64,{base64_images[0]}",
                      }
                  },
                  {
                      "type": "image_url",
                      "image_url": {
                          "url": f"data:image/png;base64,{base64_images[1]}",
                      }
                  }
              ]
        }
    )
    return msgs

######################################################################################################################    

if __name__ == "__main__":
  client = OpenAI(
      api_key=os.getenv("DASHSCOPE_API_KEY"),
      base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
  )
  folder = f'{current_file_folder}/context/4'
  files = glob.glob(f'{folder}/*.npz')
  
  base64_image, bbox_data, rot, elev = compile_expert_data(files[0])
  base64_image2, bbox_data2, rot2, elev2 = compile_expert_data(files[1])

  json_data = json_scene_data(bbox_data, rot, elev)

  completion = client.chat.completions.create(
      model="qwen2.5-vl-72b-instruct",   #"qwen-vl-max",
      messages=return_prompt_guess([base64_image, base64_image2], "pink ball", json_data)
  )
  print(completion.model_dump_json())