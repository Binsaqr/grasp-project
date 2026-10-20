"""Built-in embodiment plugins.

These plugins provide camera-free contact-set validation and metric selection.
They are not production robot controllers.
"""

from grasp_project.embodiment.plugins.parallel_jaw import ParallelJawPlugin
from grasp_project.embodiment.plugins.articulated_two_finger import ArticulatedTwoFingerPlugin
from grasp_project.embodiment.plugins.three_finger import ThreeFingerPlugin
from grasp_project.embodiment.plugins.multi_finger_hand import MultiFingerHandPlugin
from grasp_project.embodiment.plugins.human_hand import HumanHandPlugin
from grasp_project.embodiment.plugins.vacuum_suction import VacuumSuctionPlugin
from grasp_project.embodiment.plugins.soft_gripper import SoftGripperPlugin
from grasp_project.embodiment.plugins.custom import CustomPlugin

__all__ = [
    "ParallelJawPlugin",
    "ArticulatedTwoFingerPlugin",
    "ThreeFingerPlugin",
    "MultiFingerHandPlugin",
    "HumanHandPlugin",
    "VacuumSuctionPlugin",
    "SoftGripperPlugin",
    "CustomPlugin",
]
