import numpy as np
from typing import List
import sys
from .img_bbox_utils import obb_collision_expanded, SceneObject
from Levenshtein import distance

class SceneConstraints:
    """
    Class to determine spatial relationships between objects in a 3D space.
    Includes methods for checking relative positioning and determining if an object is between others.
    """
    def __init__(self, threshold: float = 0.01):
        """Initialize the SceneConstraints object with a specified threshold."""
        self.threshold = threshold
    
    def _get_coordinates(self, inst):
        """Extracts the center coordinates of the given instance."""
        return np.array(inst.center)
    
    def _get_highest_point(self, inst):
        """Extracts the highest point of the given instance."""
        corners = inst.bbox_3d
        return max(corners, key=lambda corner: corner[2])
    
    def _get_lowest_point(self, inst):
        """Extracts the lowest point of the given instance."""
        corners = inst.bbox_3d
        return min(corners, key=lambda corner: corner[2])
    
    def is_below(self, inst_0, inst_1, ratio_xy = 0.6, height_threshold = 0.0491) -> bool:
        """Checks if inst_0 is below inst_1 within the defined threshold."""
        c0, c1 = self._get_coordinates(inst_0), self._get_coordinates(inst_1)
        dx, dy = c1[:2] - c0[:2]
        dxy = np.sqrt(dx**2 + dy**2)
        dz_min = self._get_highest_point(inst_0)[2] - self._get_lowest_point(inst_1)[2]
        dz_max = - self._get_lowest_point(inst_0)[2] + self._get_highest_point(inst_1)[2]
        l0, w0, h0 = inst_0.bbox_dims
        l1, w1, h1 = inst_1.bbox_dims
        wl_threshold = (max(w0, w1) + max(l0, l1)) / 2 * ratio_xy
        if dxy < wl_threshold:
            print(f"Below relation between: {inst_0.label}, {inst_1.label}:")
            print(f"z differece: {dz_min}")
            if dz_min < height_threshold:
                return True
        return False
        
    def _check_relative_position_xy(self, inst_0, inst_1) -> bool:
        """Generalized method to check relative positioning based on x and z directionality."""
        c0, c1 = self._get_coordinates(inst_0), self._get_coordinates(inst_1)
        return c0[1] < c1[1]
    
    def _check_relative_position_z(self, inst_0, inst_1, sign=1) -> bool:
        """Generalized method to check relative positioning based on x and z directionality."""
        c0, c1 = self._get_highest_point(inst_0), self._get_highest_point(inst_1)
        return c0[2]*sign < c1[2]*sign - self.threshold
    
    def is_low(self, inst_0, inst_1) -> bool:
        """Checks if inst_0 is to the left and below inst_1."""
        return self._check_relative_position_z(inst_0, inst_1)
    
    def is_high(self, inst_0, inst_1) -> bool:
        """Checks if inst_0 is to the left and above inst_1."""
        return self._check_relative_position_z(inst_0, inst_1, sign=-1)
    
    def is_left(self, inst_0, inst_1) -> bool:
        """Checks if inst_0 is to the left and above inst_1."""
        return self._check_relative_position_xy(inst_0, inst_1)
    
    def is_between_line(self, inst_0, inst_a, inst_b, threshold: float = 0.8) -> tuple[bool, float]:
        """Determines if inst_0 is between inst_a and inst_b along a line."""
        c0, cA, cB = self._get_coordinates(inst_0), self._get_coordinates(inst_a), self._get_coordinates(inst_b)
        diff_0A, diff_B0 = c0 - cA, cB - c0
        diff_0A, diff_B0 = diff_0A[:2], diff_B0[:2] # ignore z-axis
        #calculate cosine of the angle between the two vectors
        cos_theta = np.dot(diff_0A, diff_B0) / (np.linalg.norm(diff_0A) * np.linalg.norm(diff_B0))
        # print(f"for {inst_0.label}, cos_theta between {inst_a.label} and {inst_b.label}: {cos_theta}")

        return cos_theta > threshold # cos(30) = 0.866
    
    def print_distance_xy(self, inst_0, inst_1):
        """Calculates the Euclidean distance between two instances."""
        return np.linalg.norm(self._get_coordinates(inst_0)[:2] - self._get_coordinates(inst_1)[:2])
    
    def identify_relation(self, inst_0, inst_1) -> List[str]:
        """Identifies the spatial relation between inst_0 and inst_1, considering nearby objects."""
        relations = []
        if self.is_below(inst_0, inst_1):
            relations.append(f"{inst_0.label} is below {inst_1.label}")
            return relations
        
        # left right
        relation = "left" if self.is_left(inst_0, inst_1) else "right"
        # low high
        for lh in ("low", "high"):
            if getattr(self, f"is_{lh.lower()}")(inst_0, inst_1):
                relation += lh
        
        relations.append(f"is {relation} to {inst_1.label}")
        return relations
        
    def identify_between_relations(self, inst_0, related_objs) -> List[str]:
        relations = []
        if len(related_objs) >= 2:
            # Check if inst_0 is between two other objects
            sorted_objs = sorted(related_objs, key=lambda obj: self.print_distance_xy(inst_0, obj))
            for i in range(0, len(sorted_objs) - 1):
                inst_a = sorted_objs[i]
                for j in range(i + 1, len(sorted_objs)):
                    inst_b = sorted_objs[j]
                    if self.is_between_line(inst_0, inst_a, inst_b, threshold=0.7):
                        relations.append(f"is between {inst_a.label} and {inst_b.label}")
        return relations


def compile_relation(scene_objects, 
                     bbox_expansion=0.0):
    for curr_obj in scene_objects:
        curr_obj.relations = []
        related_objs = []

        for obj in scene_objects:
            # bboxes of the two objects are colliding 
            if obj.label != curr_obj.label and obb_collision_expanded(curr_obj.bbox_3d, obj.bbox_3d, bbox_expansion):
                related_objs.append(obj)

        if not len(related_objs):
            continue
        
        checker = SceneConstraints(threshold=0.05)
        for robj in related_objs:
            curr_obj.relations += checker.identify_relation(curr_obj, robj)
        curr_obj.relations += checker.identify_between_relations(curr_obj, related_objs)

        print(f"[VLM]: Relations for {curr_obj.label}: {curr_obj.relations}")


def match_sentences(hypotheses, references):
    """
    Matches each hypothesis in A to the best reference in B using manually computed Levenshtein distance.
    Ensures 1-to-1 mapping with improved accuracy.

    Args:
        hypotheses (list of str): The generated phrases (A).
        references (list of str): The actual target phrases (B).

    Returns:
        list: List of indexes corresponding to the best match in B for each sentence in A,
              or -1 if no match is found.
    """
    best_matches = [-1] * len(hypotheses)  # Initialize matches as -1
    used_indices = set()  # Track used references

    for a_idx, hyp in enumerate(hypotheses):
        best_index = -1
        min_distance = float('inf')  # Initialize with a large number

        for b_idx, ref in enumerate(references):
            if b_idx in used_indices:  # Ensure 1-to-1 mapping
                continue

            # Remove numbers and extra characters to improve matching accuracy
            hyp_clean = ''.join([c for c in hyp if not c.isdigit()]).strip()
            ref_clean = ''.join([c for c in ref if not c.isdigit()]).strip()

            # Compute Levenshtein distance on cleaned text
            dist = distance(hyp_clean, ref_clean)

            if dist < min_distance:
                min_distance = dist
                best_index = b_idx

        if best_index != -1 and min_distance < len(hyp) * 0.5:  # Threshold to avoid incorrect matches
            best_matches[a_idx] = best_index
            used_indices.add(best_index)  # Mark this reference as used

    return best_matches

# A = ['yellow banana 1', 'blue rubik cube 1', 'orange drill 1', 'green iron 1', 'cube 1', 'white socket 1']
# B = ['orange drill', 'blue rubik"se', 'yellow banana', 'gon', 'white socket']

# match_indecies = match_sentences(A, B)
# print(match_indecies)