
"""
Subclasses for robosuite environments to define tasks that are compatible with teleoperation.
This is a convenient place to model different tasks we'd like to collect data on.

NOTE: most of these just override the intiial robot configuration to make it convenient for
teleoperation.
"""
import numpy as np

import robosuite
from robosuite.environments.sawyer_lift import *
from robosuite.environments.sawyer_push import *
from robosuite.environments.sawyer_pick_place import *
from robosuite.environments.sawyer_nut_assembly import *
from robosuite.environments.sawyer_fit import *
from robosuite.environments.sawyer_lego import *
from robosuite.environments.sawyer_assembly import *
from robosuite.environments.sawyer_clutter import *
from robosuite.environments.sawyer_door import *
from robosuite.environments.sawyer_pole import *
from robosuite.environments.sawyer_coffee import *
from robosuite.environments.sawyer_playtable import *

DEFAULT_JPOS = np.array([0.00, -1.18, 0.00, 2.18, 0.00, 0.57, 1.5708])
PUSH_JPOS = np.array([-0.40451, -0.65964, 0.09147, 2.20431, -1.19175, 0.07469, 2.59081])

### Lift Tasks ###

class SawyerLiftTeleop(SawyerLift):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLiftPositionTeleop(SawyerLiftPosition):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLiftWidePositionInitTeleop(SawyerLiftWidePositionInit):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLiftWideInitTeleop(SawyerLiftWideInit):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLiftSmallGridTeleop(SawyerLiftSmallGrid):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLiftLargeGridTeleop(SawyerLiftLargeGrid):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLiftPositionTargetTeleop(SawyerLiftPositionTarget):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPositionTargetPressTeleop(SawyerPositionTargetPress):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPositionTargetTeleop(SawyerPositionTarget):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPositionTargetRandomTeleop(SawyerPositionTargetRandom):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPositionPressTeleop(SawyerPositionPress):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS


### PT ###
class SawyerPTLGPTeleop(SawyerPTLGP):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

### Push Tasks ###

class SawyerPushTeleop(SawyerPush):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = PUSH_JPOS

class SawyerPushPositionTeleop(SawyerPushPosition):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = PUSH_JPOS

class SawyerPushPuckTeleop(SawyerPushPuck):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = PUSH_JPOS

class SawyerPushWideBarTeleop(SawyerPushWideBar):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = PUSH_JPOS

class SawyerPushLongBarTeleop(SawyerPushLongBar):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = PUSH_JPOS


### Nut Assembly Tasks ###

class SawyerNutAssemblyTeleop(SawyerNutAssembly):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblySingleTeleop(SawyerNutAssemblySingle):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblySquareTeleop(SawyerNutAssemblySquare):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblyRoundTeleop(SawyerNutAssemblyRound):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblySquareConstantRotationTeleop(SawyerNutAssemblySquareConstantRotation):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerNutAssemblySquareConstantRotationPositionTeleop(SawyerNutAssemblySquareConstantRotationPosition):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS


### Pick Place Tasks ###

class SawyerPickPlaceTeleop(SawyerPickPlace):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceSingleTeleop(SawyerPickPlaceSingle):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceMilkTeleop(SawyerPickPlaceMilk):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceBreadTeleop(SawyerPickPlaceBread):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceCerealTeleop(SawyerPickPlaceCereal):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPickPlaceCanTeleop(SawyerPickPlaceCan):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS


### Fitting Tasks ###

class SawyerFitTeleop(SawyerFit):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerFitPushLongBarTeleop(SawyerFitPushLongBar):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerFitPegInHoleTeleop(SawyerFitPegInHole):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerThreadingTeleop(SawyerThreading):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerThreadingPreciseTeleop(SawyerThreadingPrecise):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerThreadingRingTeleop(SawyerThreadingRing):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCircusTeleop(SawyerCircus):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCircusTestTeleop(SawyerCircusTest):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCircusEasyTeleop(SawyerCircusEasy):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCircusEasyTestTeleop(SawyerCircusEasyTest):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCoffeeTeleop(SawyerCoffee):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCoffeeFTTeleop(SawyerCoffeeFT):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCoffeeContactTeleop(SawyerCoffeeContact):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCoffeeContactPenaltyTeleop(SawyerCoffeeContactPenalty):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCoffeeMinimalTeleop(SawyerCoffeeMinimal):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCoffeeMinimal2Teleop(SawyerCoffeeMinimal2):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerCoffeeMinimalContactTeleop(SawyerCoffeeMinimalContact):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLegoTeleop(SawyerLego):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLegoEasyTeleop(SawyerLegoEasy):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerLegoFitTeleop(SawyerLegoFit):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerAssemblyTeleop(SawyerAssembly):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerClutterTeleop(SawyerClutter):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerDoorTeleop(SawyerDoor):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS

class SawyerPoleTeleop(SawyerPole):
    def _load_model(self):
        super()._load_model()
        # override initial joint position placement
        self.init_qpos = DEFAULT_JPOS


