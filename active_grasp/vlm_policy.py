import time
import numpy as np
from .policy import MultiViewPolicy
from .timer import Timer
from .nbv import get_voxel_at, raycast
from .spatial import SpatialTransform
from vmf_contact_main.train import main_module, parse_args_from_yaml
from scipy.spatial.transform import Rotation
import os
import numpy as np
import time
from .vlm_utils.vlm_agent import *
from .vlm_utils.nbv_generator import *
import time
import torch.multiprocessing as multiprocessing
from functools import partial
import open3d as o3d
from .vlm_utils.vlm_relation import SceneConstraints
multiprocessing.set_start_method('spawn', force=True)

current_file_folder = os.path.dirname(os.path.abspath(__file__))
O_SIZE = .3


class VLMPolicy(MultiViewPolicy):
    def __init__(self, target_object, pcd_center, min_z_dist):
        super().__init__()
        self.max_views = 10
        self.score_th = 0.5
        self.grasp_agent = main_module(parse_args_from_yaml(), learning=False)
        self.grasp_buffer = self.grasp_agent.grasp_buffer
        self.target_object_final = target_object
        # initialize the agent process
        self.vlm_agent = VLMAgent(target_object=target_object)
        self.pcd_center = pcd_center
        self.min_z_dist = min_z_dist
        self.activate()        


    def activate(self):
        self.nbv_fields = []
        self.nbv_occ = []
        self.nbv_reached = False
        self.grasp_buffer.clear()
        self.vlm_agent.clear()
        self.views = 0
        self.best_grasp = None
        self.x_d = None
        self.target_object_curr = None
        self.done = False
        self.i = 0


    def generate_view(self, img, pose, pcd_raw, rotation_angle, elevation_angle):
        # start VLM inference
        self.vlm_agent(img, 
                        pose, 
                        pcd_raw, 
                        rotation_angle, 
                        elevation_angle, 
                        self.target_object_curr_label)

        self.run_next_view(pose)
    

    def run_next_view(self, pose):
        if len(self.vlm_agent.ordered_grasp_list) > 0:
            self.target_object_curr = self.vlm_agent.ordered_grasp_list[0]
            self.ordered_grasp_list.clear()

        if self.word_based_matching() is not None:
            while True:
                # find the target object and its relations
                print(f"[Policy]: Analyzing current target object: {self.target_object_curr_label} ...")
                # match the target object label with the corresponding scene object
                self.target_object_curr: SceneObject = self.word_based_matching()
                if self.compile_related_objects(pose) is None:
                    break                
            print(f"[Policy]: Target object to grasp: {self.target_object_curr_label} ...")
            # create the NBV fields from all tall objects in the scene
            self.create_nbv_field_and_wait(pose)
            self.vlm_agent.set_vlm_cmd("scene")
        else:
            if len(self.scene_objects) == 0:
                print(f"[Policy]: No objects are found in the scene, the camera observes the empty scene.")
                self.vlm_agent.set_vlm_cmd("scene")
            else:
                print(f"[Policy]: Target object {self.target_object_curr_label} is not found, start guessing ...")
                self.vlm_agent.set_vlm_cmd("guess")
    

    def compile_related_objects(self, pose):
        # nbv_fields = []
        current_cam_pos = np.array(pose.translation)
        # Decision making based on the relations
        if len(self.target_object_curr.relations) > 0:
            for rel in self.target_object_curr.relations:
                if "below" in rel:
                    obj = rel.split("below ")[1]
                    obj:SceneObject = self.scene_objects[obj]
                    print(f"[Policy]: Relation: below {obj.label}")

                    # Rule 0: Uncovers the object above the target object
                    print(f"[Policy]: Before grasping the object: {self.target_object_curr_label}, object: {obj.label} needs to be grasped")
                    # update the target object
                    self.target_object_curr = obj
                    return True
                    
                elif "between" in rel:
            
                    # extract the objects from the relation
                    obj1, obj2 = rel.split("between ")[1].split(" and ")
                    obj1, obj2 = self.scene_objects[obj1], self.scene_objects[obj2]
                    print(f"[Policy]: Relation: between {obj1.label} and {obj2.label}")

                    # distance to the camera
                    cam_obj1_dist = np.linalg.norm(current_cam_pos - obj1.center)
                    cam_obj2_dist = np.linalg.norm(current_cam_pos - obj2.center)
                    cam_target_object_dist = np.linalg.norm(current_cam_pos - self.target_object_curr.center)
                    
                    # whether low to both objects
                    obj_1_high = any(f"low to {obj1.label}" in rel for rel in self.target_object_curr.relations)
                    obj_2_high = any(f"low to {obj2.label}" in rel for rel in self.target_object_curr.relations)

                    # Rule 1: Uncovers the close object in front of camera if both objects are high
                    if obj_1_high and obj_2_high:
                        # update the target object
                        if cam_obj1_dist < cam_obj2_dist:
                            print(f"[Policy]: Before grasping the object: {self.target_object_curr_label}, object: {obj1.label} needs to be grasped")
                            self.target_object_curr=obj1
                        else:
                            print(f"[Policy]: Before grasping the object: {self.target_object_curr_label}, object: {obj2.label} needs to be grasped")
                            self.target_object_curr=obj2
                        return True

                    # IMPORTANT !!!  Rule 2&3 is not considered since NBV field will be generated by all high objects in the scene
                    # # Rule 2: NBV is across the perpendicular plane between the target object and the high object
                    # elif not obj_2_high and obj_1_high:
                    #     if cam_obj2_dist < cam_target_object_dist:
                    #         nbv_fields.append(self.nbv_func_center(obj2))
                    # elif not obj_2_high and obj_1_high:
                    #     if cam_obj1_dist < cam_target_object_dist:
                    #         nbv_fields.append(self.nbv_func_center(obj1))
                            
                # elif "to" in rel:
                #     relative_pos = rel[3:].split(" to")[0]
                #     obj = rel.split(" to ")[1]
                #     obj = self.scene_objects[obj]
                #     print(f"[Policy]: Relation: {relative_pos} to {obj.label}")
                #     if not "high" in relative_pos:

                #         # Rule 3: NBV is across the perpendicular plane between the target object 
                #         # and the object and towards the target object
                #         nbv_fields.append(self.nbv_func_center(obj))
        else:
            print(f"[Policy]: Object {self.target_object_curr_label} has no relations")
        
        # self.nbv_fields = nbv_fields

    def create_nbv_field_and_wait(self, pose):
        nbv_fields = []
        self.nbv_occ = []
        for obj in self.scene_objects.values():
            if obj.label != self.target_object_curr_label: 
                # all non-high objects are considered for NBV
                if not SceneConstraints().is_high(self.target_object_curr, obj):
                    print(f"[Policy]: Object {obj.label} considered for NBV")
                    nbv_fields.append(self.nbv_func_center(obj))
                    nbv_fields+=self.nbv_func_bbox(obj)
        self.nbv_fields = nbv_fields

        # visualize the NBV fields
        self.visualize_nbv_fields(pose.translation)

        if len(self.nbv_fields) > 0:
            self.wait_nbv_run()
        else:
            print(f"[Policy]: No NBV fields are generated.")

    def nbv_func_center(self, obj):
        self.nbv_occ.append(obj.center)
        return partial(
            query_tangent_vector,
            S = self.pcd_center,
            R_s = self.min_z_dist,
            P1 = obj.center,
            P2 = self.target_object_curr.center
        )  

    def nbv_func_bbox(self, obj):
        self.nbv_occ += [pt for pt in obj.bbox_3d]
        return [partial(
            query_tangent_vector,
            S = self.pcd_center,
            R_s = self.min_z_dist,
            P1 = point,
            P2 = self.target_object_curr.center
        )  for point in obj.bbox_3d]   
    
    def wait_nbv_run(self):
        time_curr = time.time()
        while not self.nbv_reached:
            # wait for the NBV fields to be generated
            print(f"[Policy]: Waiting for the NBV to be reached ...")
            if time.time() - time_curr > 20:
                break
            time.sleep(1)
        print(f"[Policy]: NBV is reached.")
        self.views += 1
    
    def visualize_nbv_fields(self, current_cam_pos):
        if True and len(self.nbv_fields) > 0:
            current_cam_pos = np.array(current_cam_pos)
            # visualize the NBV fields
            animate_query_tangent_vector(self.pcd_center, 
                                        self.min_z_dist, 
                                        current_cam_pos,
                                        self.target_object_curr.center, 
                                        self.nbv_occ, 
                                        self.nbv_fields)
    

    def word_based_matching(self):
        for obj_id in self.scene_objects.keys():
            if all(word in obj_id for word in self.target_object_curr_label.split()):
                target_obj = self.scene_objects[obj_id]
                # print(f"[Policy]: Target object: {target_obj.label} at {target_obj.center} is found")
                return target_obj
        print(f"[Policy]: Object {self.target_object_curr_label} is not found")
        return None
    

    def update(self, img, depth, pcd_raw, x, rotation_angle, elevation_angle):
        x = self.translate_pose(x)

        with Timer("view_generation"):
            self.generate_view(img, x, pcd_raw, rotation_angle, elevation_angle)
        
        if self.views > self.max_views or self.best_grasp_prediction_is_stable():
            self.done = True
    

    def update_grasp(self, pcd_raw, interactive_vis=False):
        if not self.done and pcd_raw.shape[0] > 0:
            pcd = self.denoise_pcd(pcd_raw)

            # print("Processed point cloud: ", pcd.shape)
                
            with Timer("grasp_prediction"):
                time_curr = time.time()
                self.curr_grasp = self.grasp_agent.inference(pcd, 
                                        pcd_from_prompt=None,
                                        pcd_shift=self.pcd_center,
                                        graspness_th=self.score_th,
                                        grasp_height_th=5e-3,
                                        vis=True,
                                        interactive_vis=interactive_vis)
                self.i += 1
                # print(f"[vMF-Contact] Time taken for grasp inference: {time.time() - time_curr}")


    def best_grasp_prediction_is_stable(self, sort_by="kappa"):
        if self.target_object_curr is not None:
                # get the current pcd of the target object
            pcd_from_prompt=self.target_object_curr.pcd
            
            # get the best grasp prediction on the target object
            self.best_grasp = self.grasp_buffer.get_pose_fused_best(sort_by=sort_by, pcd_from_prompt=pcd_from_prompt, sample_num=3)

            if self.best_grasp is None:
                print(f"[vMF-Contact]: No grasp prediction on object: {self.target_object_curr_label}, even if it's found.")
                self.vlm_agent.set_vlm_cmd("ordered_grasp")
                return False
            
            print(f"[vMF-Contact]: Best grasp prediction on object: {self.target_object_curr_label}, identified.")
            return True
        
        print(f"[vMF-Contact]: No target object found on object: {self.target_object_curr_label}")
        return False
    

    def translate_pose(self, x):
        pos, quat = ([x.position.x, 
                      x.position.y, 
                      x.position.z], 
                      [x.orientation.x, 
                       x.orientation.y, 
                       x.orientation.z, 
                       x.orientation.w])
        x = SpatialTransform.from_translation(pos)
        x.rotation = Rotation.from_quat(quat)
        return x
    

    def denoise_pcd(self, pcd):
        pcd = pcd - self.pcd_center
        pcd = pcd[(pcd[:, 0] > -O_SIZE) & (pcd[:, 0] < O_SIZE)]
        pcd = pcd[(pcd[:, 1] > -O_SIZE) & (pcd[:, 1] < O_SIZE)]
        pcd = pcd[(pcd[:, 2] > 0.025) & (pcd[:, 2] < 0.45)]
        # pcd = denoise_point_cloud(pcd, method="statistical", nb_neighb.35ors=20, std_ratio=0.1)
        # pcd_vis = o3d.geometry.PointCloud()
        # pcd_vis.points = o3d.utility.Vector3dVector(pcd)
        # o3d.visualization.draw_geometries([pcd_vis])
        return pcd
    
    def query_field_fusion_from_list(self, translation):
        assert len(self.nbv_fields) > 0, "No NBV fields to query."
        translation = np.array(translation)
        return query_tangent_vector_sum_from_field_list(S=self.pcd_center,
                                                        P_q = translation,
                                                        field_list=self.nbv_fields)
    
    @property
    def target_object_curr_label(self):
        if self.target_object_curr is not None:
            return self.target_object_curr.label
        return self.target_object_final
    
    @property
    def scene_objects(self):
        return self.vlm_agent.scene_objects
    
    @property
    def ordered_grasp_list(self) -> list[SceneObject]:
        return self.vlm_agent.ordered_grasp_list