from .policy import register
from .baselines import *
from .vlm import VLMPolicy
from .nbv import NextBestView

register("initial-view", InitialView)
register("top-view", TopView)
register("top-trajectory", TopTrajectory)
register("fixed-trajectory", FixedTrajectory)
register("nbv", NextBestView)
register("vlm", VLMPolicy)
