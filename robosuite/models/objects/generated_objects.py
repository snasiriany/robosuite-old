import random
import numpy as np
import xml.etree.ElementTree as ET
from copy import deepcopy

import robosuite.utils.transform_utils as T
from robosuite.models.objects import MujocoGeneratedObject, MujocoXMLObject
from robosuite.utils.mjcf_utils import new_body, new_geom, new_site, new_joint, array_to_string
from robosuite.utils.mjcf_utils import RED, GREEN, BLUE

from robosuite.models.objects import CoffeeMachineBodyObject, CoffeeMachineLidObject, CoffeeMachineBaseObject, CoffeeMachinePodObject


class PotWithHandlesObject(MujocoGeneratedObject):
    """
    Generates the Pot object with side handles (used in BaxterLift)
    """

    def __init__(
        self,
        body_half_size=None,
        handle_radius=0.01,
        handle_length=0.09,
        handle_width=0.09,
        rgba_body=None,
        rgba_handle_1=None,
        rgba_handle_2=None,
        solid_handle=False,
        thickness=0.025,  # For body
    ):
        super().__init__()
        if body_half_size:
            self.body_half_size = body_half_size
        else:
            self.body_half_size = np.array([0.07, 0.07, 0.07])
        self.thickness = thickness
        self.handle_radius = handle_radius
        self.handle_length = handle_length
        self.handle_width = handle_width
        if rgba_body:
            self.rgba_body = np.array(rgba_body)
        else:
            self.rgba_body = RED
        if rgba_handle_1:
            self.rgba_handle_1 = np.array(rgba_handle_1)
        else:
            self.rgba_handle_1 = GREEN
        if rgba_handle_2:
            self.rgba_handle_2 = np.array(rgba_handle_2)
        else:
            self.rgba_handle_2 = BLUE
        self.solid_handle = solid_handle

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.body_half_size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.body_half_size[2]])

    def get_horizontal_radius(self):
        return np.sqrt(2) * (max(self.body_half_size) + self.handle_length)

    @property
    def handle_distance(self):
        return self.body_half_size[1] * 2 + self.handle_length * 2

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        for geom in five_sided_box(
            self.body_half_size, self.rgba_body, 1, self.thickness
        ):
            main_body.append(geom)
        handle_z = self.body_half_size[2] - self.handle_radius
        handle_1_center = [0, self.body_half_size[1] + self.handle_length, handle_z]
        handle_2_center = [
            0,
            -1 * (self.body_half_size[1] + self.handle_length),
            handle_z,
        ]
        # the bar on handle horizontal to body
        main_bar_size = [
            self.handle_width / 2 + self.handle_radius,
            self.handle_radius,
            self.handle_radius,
        ]
        side_bar_size = [self.handle_radius, self.handle_length / 2, self.handle_radius]
        handle_1 = new_body(name="handle_1")
        if self.solid_handle:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1",
                    pos=[0, self.body_half_size[1] + self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
        else:
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_c",
                    pos=handle_1_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )
            handle_1.append(
                new_geom(
                    geom_type="box",
                    name="handle_1_-",
                    pos=[
                        -self.handle_width / 2,
                        self.body_half_size[1] + self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_1,
                    group=1,
                )
            )

        handle_2 = new_body(name="handle_2")
        if self.solid_handle:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2",
                    pos=[0, -self.body_half_size[1] - self.handle_length / 2, handle_z],
                    size=[
                        self.handle_width / 2,
                        self.handle_length / 2,
                        self.handle_radius,
                    ],
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
        else:
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_c",
                    pos=handle_2_center,
                    size=main_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_+",  # + for positive x
                    pos=[
                        self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )
            handle_2.append(
                new_geom(
                    geom_type="box",
                    name="handle_2_-",
                    pos=[
                        -self.handle_width / 2,
                        -self.body_half_size[1] - self.handle_length / 2,
                        handle_z,
                    ],
                    size=side_bar_size,
                    rgba=self.rgba_handle_2,
                    group=1,
                )
            )

        main_body.append(handle_1)
        main_body.append(handle_2)
        main_body.append(
            new_site(
                name="pot_handle_1",
                rgba=self.rgba_handle_1,
                pos=handle_1_center - np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(
            new_site(
                name="pot_handle_2",
                rgba=self.rgba_handle_2,
                pos=handle_2_center + np.array([0, 0.005, 0]),
                size=[0.005],
            )
        )
        main_body.append(new_site(name="pot_center", pos=[0, 0, 0], rgba=[1, 0, 0, 0]))

        return main_body

    def handle_geoms(self):
        return self.handle_1_geoms() + self.handle_2_geoms()

    def handle_1_geoms(self):
        if self.solid_handle:
            return ["handle_1"]
        return ["handle_1_c", "handle_1_+", "handle_1_-"]

    def handle_2_geoms(self):
        if self.solid_handle:
            return ["handle_2"]
        return ["handle_2_c", "handle_2_+", "handle_2_-"]

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


def five_sided_box(size, rgba, group, thickness):
    """
    Args:
        size ([float,flat,float]):
        rgba ([float,float,float,float]): color
        group (int): Mujoco group
        thickness (float): wall thickness

    Returns:
        []: array of geoms corresponding to the
            5 sides of the pot used in BaxterLift
    """
    geoms = []
    x, y, z = size
    r = thickness / 2
    geoms.append(
        new_geom(
            geom_type="box", size=[x, y, r], pos=[0, 0, -z + r], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, -y + r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[x, r, z], pos=[0, y - r, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[x - r, 0, 0], rgba=rgba, group=group
        )
    )
    geoms.append(
        new_geom(
            geom_type="box", size=[r, y, z], pos=[-x + r, 0, 0], rgba=rgba, group=group
        )
    )
    return geoms


DEFAULT_DENSITY_RANGE = [200, 500, 1000, 3000, 5000]
DEFAULT_FRICTION_RANGE = [0.25, 0.5, 1, 1.5, 2]


def _get_size(size,
              size_max,
              size_min,
              default_max,
              default_min):
    """
        Helper method for providing a size,
        or a range to randomize from
    """
    if len(default_max) != len(default_min):
        raise ValueError('default_max = {} and default_min = {}'
                         .format(str(default_max), str(default_min)) +
                         ' have different lengths')
    if size is not None:
        if (size_max is not None) or (size_min is not None):
            raise ValueError('size = {} overrides size_max = {}, size_min = {}'
                             .format(size, size_max, size_min))
    else:
        if size_max is None:
            size_max = default_max
        if size_min is None:
            size_min = default_min
        size = np.array([np.random.uniform(size_min[i], size_max[i])
                         for i in range(len(default_max))])
    return size


def _get_randomized_range(val,
                          provided_range,
                          default_range):
    """
        Helper to initialize by either value or a range
        Returns a range to randomize from
    """
    if val is None:
        if provided_range is None:
            return default_range
        else:
            return provided_range
    else:
        if provided_range is not None:
            raise ValueError('Value {} overrides range {}'
                             .format(str(val), str(provided_range)))
        return [val]


class CompositeBodyObject(MujocoGeneratedObject):
    """
    An object constructed out of basic objects to make more intricate shapes.
    """

    def __init__(
        self,
        objects,
        total_size,
        object_locations,
        joint=None,
        locations_relative_to_center=False,
        object_quats=None,
    ):
        """
        Args:
            objects (list): list of MujocoGeneratedObject instances

            total_size (list): half-size in each dimension for the bounding box for
                this CompositeBody object

            object_locations (list): list of object locations in the composite. Each 
                location should be a list or tuple of 3 elements and all 
                locations are relative to the lower left corner of the total box 
                (e.g. (0, 0, 0) corresponds to this corner).

            joint (list): list of joints to add for the entire composite object
        """
        super().__init__(joint=joint, rgba=None)

        assert np.all([isinstance(elem, MujocoGeneratedObject) or isinstance(elem, MujocoXMLObject) for elem in objects])
        self.objects = objects
        self.total_size = np.array(total_size)

        self.object_locations = np.array(object_locations)
        assert self.object_locations.shape[0] == len(self.objects)

        self.locations_relative_to_center = locations_relative_to_center
        self.object_quats = deepcopy(object_quats) if object_quats is not None else None

        # merge assets of objects
        for obj in self.objects:
            for asset in obj.asset:
                asset_name = asset.get("name")
                asset_type = asset.tag
                # Avoids duplication
                pattern = "./{}[@name='{}']".format(asset_type, asset_name)
                if self.asset.find(pattern) is None:
                    self.asset.append(asset)

    def get_bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    def get_top_offset(self):
        return np.array([0., 0., self.total_size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)

    def get_bounding_box_size(self):
        return np.array(self.total_size)

    def _get_body(self, name=None, site=None, visual=False):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        # give main body a small mass in order to have a free joint
        inertial = ET.Element("inertial")
        inertial.set("pos", "0 0 0")
        inertial.set("mass", "0.0001")
        inertial.set("diaginertia", "0.0001 0.0001 0.0001")
        main_body.append(inertial)

        for i in range(self.object_locations.shape[0]):

            # body name
            if name is None:
                body_name = "body_{}".format(i)
            else:
                body_name = "{}_{}".format(name, i)

            # bounding box for object
            cartesian_size = self.objects[i].get_bounding_box_size()

            if self.locations_relative_to_center:
                # no need to convert
                pos = self.object_locations[i]
            else:
                # use object location to convert to position coordinate (the origin is the
                # center of the composite object)
                loc = self.object_locations[i]
                pos = [
                    (-self.total_size[0] + cartesian_size[0]) + loc[0],
                    (-self.total_size[1] + cartesian_size[1]) + loc[1],
                    (-self.total_size[2] + cartesian_size[2]) + loc[2],
                ]

            # add object body
            if visual:
                obj_body = self.objects[i].get_visual(
                    name=body_name,
                    site=site,
                )
            else:
                obj_body = self.objects[i].get_collision(
                    name=body_name,
                    site=site,
                )

            # set body position
            obj_body.set("pos", array_to_string(pos))

            if self.object_quats is not None:
                obj_body.set("quat", array_to_string(self.object_quats[i]))

            # add object joints
            for j, joint in enumerate(self.objects[i].joint):
                obj_body.append(new_joint(name="{}_{}".format(body_name, j), **joint))
            main_body.append(obj_body)

        # add site if requested
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))

        return main_body

    def get_collision(self, name=None, site=None):
        return self._get_body(name=name, site=site, visual=False)

    def get_visual(self, name=None, site=None):
        return self._get_body(name=name, site=site, visual=True)


class HingeStackObject(CompositeBodyObject):
    def __init__(self):
        box1 = BoxObject(
            size=[0.02, 0.02, 0.02],
            rgba=[1, 0, 0, 1],
            joint=[],
            density=5000.,
        )

        # allow for rotation about the edge of the cube
        hinge_joint = dict(
            type="hinge",
            axis="1 0 0",
            pos="0 -0.02 -0.02",
            limited="true",
            range="0 1.57",
            # range="0 0.7",
            # frictionloss="0.1", 
            # damping="0.01", 
            # springref="0",
            # stiffness="0.1",
        )
        # hinge_joint_2 = dict(
        #     type="hinge",
        #     axis="1 0 0",
        #     pos="0 -0.02 -0.02",
        #     limited="true",
        #     range="0.7 1.57",
        # )
        box2 = BoxObject(
            size=[0.02, 0.02, 0.02],
            rgba=[0, 1, 0, 1],
            density=200.,
            joint=[hinge_joint],
            # joint=[hinge_joint, hinge_joint_2],
        )

        total_size = [0.02, 0.02, 0.04]
        object_locations = [
            [0., 0., 0.], 
            [0., 0., 2 * 0.02],
        ]

        super().__init__(
            objects=[box1, box2],
            total_size=total_size,
            object_locations=object_locations,
            # joint=[],
            joint=None,
        )


class CoffeeMachineObject2(CompositeBodyObject):
    def __init__(self):

        # pieces of the coffee machine
        body = CoffeeMachineBodyObject(joint=[])
        body_size = body.get_bounding_box_size()
        body_location = [0., 0., 0.]

        lid = CoffeeMachineLidObject(joint=[])
        lid_size = lid.get_bounding_box_size()
        # add tolerance to allow lid to open fully
        lid_location = [
            body_size[0] - lid_size[0],
            2. * body_size[1] + 0.01,
            2. * (body_size[2] - lid_size[2]) + 0.005,
        ]

        # add in hinge joint to lid
        hinge_pos = [0., -lid_size[1], -lid_size[2]]
        hinge_joint = dict(
            type="hinge",
            axis="1 0 0",
            pos=array_to_string(hinge_pos),
            limited="true",
            range="0 1.57",
            # frictionloss="1.0", 
            damping="0.01",
        )
        lid = CoffeeMachineLidObject(joint=[hinge_joint])

        base = CoffeeMachineBaseObject(joint=[])
        base_size = base.get_bounding_box_size()
        base_location = [
            body_size[0] - base_size[0],
            2. * body_size[1],
            0.
        ]

        pod_holder_holder = BoxObject(
            size=[
                0.01, 
                # tolerance for having the lid stick out a little from the holder
                0.9 * (lid_size[1] - lid_size[0]), 
                0.005,
            ],
            rgba=[0.514, 0.286, 0.204, 1], # brown
            joint=[],
        )
        pod_holder_holder_size = pod_holder_holder.get_bounding_box_size()
        pod_holder_holder_location = [
            body_size[0] - pod_holder_holder_size[0],
            2. * body_size[1],
            # put right underneath lid
            2. * (body_size[2] - lid_size[2] - pod_holder_holder_size[2]),
        ]

        pod_holder = CupObject(
            outer_cup_radius=lid_size[0],
            # outer_cup_radius=0.03,
            inner_cup_radius=0.025,
            cup_height=0.025,
            cup_ngeoms=64,#8,
            cup_base_height=0.005,
            cup_base_offset=0.005,
            add_handle=False,
            rgba=[1, 0, 0, 1],
            density=100.,
            joint=[],
        )
        pod_holder_size = pod_holder.get_bounding_box_size()
        pod_holder_location = [
            body_size[0] - pod_holder_size[0],
            2. * (body_size[1] + pod_holder_holder_size[1]),
            # put right underneath lid
            2. * (body_size[2] - lid_size[2] - pod_holder_size[2])
        ]

        total_size = [
            body_size[0],
            body_size[1] + base_size[1],
            body_size[2],
        ]

        objects = [
            body,
            lid,
            base,
            pod_holder_holder,
            pod_holder,
        ]

        object_locations = [
            body_location,
            lid_location,
            base_location,
            pod_holder_holder_location,
            pod_holder_location,
        ]

        object_quats = [
            [0., 0., 0., 1.], # z-rotate body and base by 180
            [1., 0., 0., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.],
            [1., 0., 0., 0.],
        ]

        super().__init__(
            objects=objects,
            total_size=total_size,
            object_locations=object_locations,
            object_quats=object_quats,
            joint=[],
            # joint=None,
        )


class CoffeeMachineObject(CompositeBodyObject):
    def __init__(self):

        # body of the coffee machine 
        body_size = [0.04, 0.04, 0.08]
        # body = BoxObject(
        #     size=body_size,
        #     # rgba=[0.5, 0.5, 0.5, 1], # grey
        #     rgba=[0.514, 0.286, 0.204, 1], # brown
        #     density=1000.,
        #     joint=[],
        # )

        top_frac_y = 0.75
        top_frac_z = 0.25
        geom_names = [
            'top', 
            'base'
        ]
        geom_sizes = [
            [body_size[0], body_size[1] * top_frac_y, body_size[2] * top_frac_z],
            [body_size[0], body_size[1], body_size[2] * (1. - top_frac_z)],
        ]
        geom_locations = [
            [0., 0., 2. * geom_sizes[1][2]],
            [0., 0., 0.],
        ]
        body = CompositeBoxObject(
            total_size=body_size,
            geom_locations=geom_locations,
            geom_sizes=geom_sizes,
            geom_names=geom_names,
            geom_rgbas=None,
            geom_frictions=None,
            # rgba=[0.5, 0.5, 0.5, 1], # grey
            rgba=[0.514, 0.286, 0.204, 1], # brown
            density=1000.,
            joint=[],
        )

        # platform to hold mug
        platform_size = [0.03, 0.03, 0.01]
        platform = BoundingObject(
            size=platform_size,
            hole_size=[0.02, 0.02, 0.005],
            hole_location=[0., 0.],
            hole_rgba=[0., 0., 1., 1], # blue
            joint=[],
            rgba=[1., 0., 0., 1], # red
            density=1000.,
        )

        # holder for keurig pod
        pod_holder_size = [0.02, 0.02, 0.02]
        pod_holder_hole_size = [0.01, 0.01, 0.01]
        pod_holder = BoundingObject(
            size=pod_holder_size,
            hole_size=pod_holder_hole_size,
            hole_location=[0., 0.],
            hole_rgba=[0., 0., 1., 1], # blue
            joint=[],
            rgba=[1., 0., 1., 1], # purple
            density=1000.,
        )

        # size of gap between wall of base body and edge of pod holder
        pod_holder_gap = (pod_holder_size[1] - pod_holder_hole_size[1])

        # used to attach machine body to pod holder
        # pod_holder_holder_size = [body_size[0], (pod_holder_size[1] - pod_holder_hole_size[1]) / 2., 0.02]
        pod_holder_holder_size = [
            0.5 * pod_holder_size[0], 
            (1. - top_frac_y) * body_size[1] + (pod_holder_gap / 2.), 
            0.5 * pod_holder_size[2],
        ]
        pod_holder_holder = BoxObject(
            size=pod_holder_holder_size,
            rgba=[0.5, 0.5, 0.5, 1], # grey
            density=1000.,
            joint=[],
        )

        # pod cover (lid)
        tolerance = 1.03 # tolerance for fit around pod holder
        thickness = pod_holder_size[0] - pod_holder_hole_size[0] # used as thickness of bar and height of cover
        pod_cover_total_size = [
            tolerance * pod_holder_size[0] + thickness, 
            tolerance * pod_holder_size[1] + thickness / 2., 
            thickness,
        ]

        # create geoms for the top cover and the surrounding parts of the handle
        geom_names = [
            'top', 
            'front', 
            'left', 
            'right',
        ]
        geom_sizes = [
            [tolerance * pod_holder_size[0], tolerance * pod_holder_size[1], thickness / 2.],
            [tolerance * pod_holder_size[0] + thickness, thickness / 2., thickness],
            [thickness / 2., tolerance * pod_holder_size[1], thickness],
            [thickness / 2., tolerance * pod_holder_size[1], thickness],
        ]
        geom_locations = [
            [thickness, 0., thickness],
            [0., 2. * tolerance * pod_holder_size[1], 0.],
            [2. * (tolerance * pod_holder_size[0] + thickness / 2.), 0., 0.],
            [0., 0., 0.],

        ]

        hinge_pos = [0., -pod_cover_total_size[1], -pod_cover_total_size[2]]
        hinge_joint = dict(
            type="hinge",
            axis="1 0 0",
            pos=array_to_string(hinge_pos),
            limited="true",
            range="0 1.57",
            # frictionloss="1.0", 
            damping="0.01",
        )

        pod_cover = CompositeBoxObject(
            total_size=pod_cover_total_size,
            geom_locations=geom_locations,
            geom_sizes=geom_sizes,
            geom_names=geom_names,
            geom_rgbas=None,
            geom_frictions=None,
            # joint=[],
            joint=[hinge_joint],
            rgba=[0.5, 0.5, 0.5, 1], # grey
            density=1000.,
        )

        total_size = [
            body_size[0], 
            body_size[1] + pod_holder_holder_size[1] + pod_cover_total_size[1], 
            body_size[2] + thickness / 2.,
        ]
        objects = [
            body,
            platform,
            pod_holder,
            pod_holder_holder,
            pod_cover,
        ]
        object_locations = [
            [0., 0., 0.],
            [body_size[0] - platform_size[0], 2. * body_size[1], 0.],
            [body_size[0] - pod_holder_size[0], 2. * (body_size[1] + pod_holder_gap / 2.), 2. * (body_size[2] - pod_holder_size[2])],
            # pod_holder_holder z-location is center of top piece of body
            [body_size[0] - pod_holder_holder_size[0], 2. * body_size[1] * top_frac_y, 2. * body_size[2] * (1. - top_frac_z) + (body_size[2] * top_frac_z - pod_holder_holder_size[1])],
            [body_size[0] - pod_cover_total_size[0], 2. * (body_size[1] + pod_holder_gap / 2.), 2. * body_size[2] - pod_cover_total_size[2]],
        ]

        super().__init__(
            objects=objects,
            total_size=total_size,
            object_locations=object_locations,
            joint=[],
            # joint=None,
        )


class CoffeePodObject(CompositeBodyObject):
    def __init__(
        self,
        lid_radius=0.02,
        lid_height=0.005,
        lid_rgba=None,
        pod_radius=0.01,
        pod_height=0.02,
        pod_rgba=None,
        joint=None,
        density=None,
    ):
        self.lid_radius = lid_radius
        self.lid_height = lid_height
        self.lid_rgba = lid_rgba
        self.pod_radius = pod_radius
        self.pod_height = pod_height
        self.pod_rgba = pod_rgba

        self.lid = CylinderObject(
            size=[self.lid_radius, self.lid_height],
            rgba=self.lid_rgba,
            density=density,
            solref=[0.02, 1.],
            solimp=[0.998, 0.998, 0.001],
            joint=[],
        )

        self.pod = CylinderObject(
            size=[self.pod_radius, self.pod_height],
            rgba=self.pod_rgba,
            density=density,
            solref=[0.02, 1.],
            solimp=[0.998, 0.998, 0.001],
            joint=[],
        )

        # just stack the cylinders on top of each other
        objects = [self.pod, self.lid]
        total_size = [
            max(self.lid_radius, self.pod_radius),
            max(self.lid_radius, self.pod_radius),
            self.lid_height + self.pod_height,
        ]
        object_locations = [
            [total_size[0] - self.pod_radius, total_size[1] - self.pod_radius, 0.],
            [total_size[0] - self.lid_radius, total_size[1] - self.lid_radius, 2. * self.pod_height],
        ]

        super().__init__(
            objects=objects,
            total_size=total_size,
            object_locations=object_locations,
            joint=joint,
            locations_relative_to_center=False,
            object_quats=None,
        )


class CompositeObject(MujocoGeneratedObject):
    """
    An object constructed out of basic geoms to make more intricate shapes.
    """

    def __init__(
        self,
        total_size,
        geom_types,
        geom_locations,
        geom_sizes,
        geom_names=None,
        geom_rgbas=None,
        geom_frictions=None,
        joint=None,
        rgba=None,
        density=100.,
        solref=[0.02, 1.],
        solimp=[0.9, 0.95, 0.001],
        locations_relative_to_center=False,
        geom_quats=None,
    ):
        """
        Args:
            total_size (list): half-size in each dimension for the bounding box for
                this Composite object

            geom_types (list): list of geom types in the composite. Must correspond
                to MuJoCo geom primitives, such as "box" or "capsule".

            geom_locations (list): list of geom locations in the composite. Each 
                location should be a list or tuple of 3 elements and all 
                locations are relative to the lower left corner of the total box 
                (e.g. (0, 0, 0) corresponds to this corner).

            geom_sizes (list): list of geom sizes ordered the same as @geom_locations

            geom_names (list): list of geom names ordered the same as @geom_locations. The
                names will get appended with an underscore to the passed name in @get_collision
                and @get_visual

            geom_rgbas (list): list of geom colors ordered the same as @geom_locations. If 
                passed as an argument, @rgba is ignored.
        """
        super().__init__(joint=joint, rgba=rgba)

        self.total_size = np.array(total_size)
        self.geom_types = np.array(geom_types)
        self.geom_locations = np.array(geom_locations)
        self.geom_sizes = deepcopy(geom_sizes)
        self.geom_names = list(geom_names) if geom_names is not None else None
        self.geom_rgbas = list(geom_rgbas) if geom_rgbas is not None else None
        self.geom_frictions = list(geom_frictions) if geom_frictions is not None else None
        self.rgba = rgba
        self.density = density
        self.solref = list(solref)
        self.solimp = list(solimp)
        self.locations_relative_to_center = locations_relative_to_center
        self.geom_quats = deepcopy(geom_quats) if geom_quats is not None else None

    def get_bottom_offset(self):
        return np.array([0., 0., -self.total_size[2]])

    def get_top_offset(self):
        return np.array([0., 0., self.total_size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.total_size[:2], 2)

    def _size_to_cartesian_half_lengths(self, geom_type, geom_size):
        """
        converts from geom size specification to x, y, and z half-length bounding box
        """
        if geom_type in ['box', 'ellipsoid']:
            return geom_size
        if geom_type == 'sphere':
            # size is radius
            return [geom_size[0], geom_size[0], geom_size[0]]
        if geom_type == 'capsule':
            # size is radius, half-length of cylinder part
            return [geom_size[0], geom_size[0], geom_size[0] + geom_size[1]]
        if geom_type == 'cylinder':
            # size is radius, half-length
            return [geom_size[0], geom_size[0], geom_size[1]]
        raise Exception("unsupported geom type!")

    def _make_geoms(self, name=None, site=None, **geom_properties):
        main_body = new_body()
        if name is not None:
            main_body.set("name", name)

        for i in range(self.geom_locations.shape[0]):

            # geom type
            geom_type = self.geom_types[i]

            # get cartesian size from size spec
            size = self.geom_sizes[i]
            cartesian_size = self._size_to_cartesian_half_lengths(geom_type, size)

            if self.locations_relative_to_center:
                # no need to convert
                pos = self.geom_locations[i]
            else:
                # use geom location to convert to position coordinate (the origin is the
                # center of the composite object)
                loc = self.geom_locations[i]
                pos = [
                    (-self.total_size[0] + cartesian_size[0]) + loc[0],
                    (-self.total_size[1] + cartesian_size[1]) + loc[1],
                    (-self.total_size[2] + cartesian_size[2]) + loc[2],
                ]

            # geom name
            if self.geom_names is not None:
                geom_name = "{}_{}".format(name, self.geom_names[i])
            else:
                geom_name = "{}_{}".format(name, i)

            # geom rgba
            if self.geom_rgbas is not None and self.geom_rgbas[i] is not None:
                geom_rgba = self.geom_rgbas[i]
            else:
                geom_rgba = self.rgba

            # geom friction
            if self.geom_frictions is not None and self.geom_frictions[i] is not None:
                geom_friction = self.geom_frictions[i]
            else:
                geom_friction = np.array([1., 0.005, 0.0001]) # mujoco default
            geom_friction = array_to_string(geom_friction)

            if self.geom_quats is not None:
                geom_properties['quat'] = array_to_string(self.geom_quats[i])

            # add geom
            main_body.append(
                new_geom(
                    size=size, 
                    pos=pos, 
                    name=geom_name,
                    rgba=geom_rgba,
                    geom_type=geom_type,
                    friction=geom_friction,
                    **geom_properties,
                )
            )

        # add site if requested
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            template["rgba"] = "1 0 0 0"
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))

        return main_body

    def get_collision(self, name=None, site=None):
        geom_properties = {
            'group': 1,
            'density': str(self.density),
            'solref': array_to_string(self.solref),
            'solimp': array_to_string(self.solimp),
        }
        if self.rgba is None and self.geom_rgbas is None:
            # if no color, default to lego material
            geom_properties['material'] = 'lego'
        return self._make_geoms(name=name, site=site, **geom_properties)

    def get_visual(self, name=None, site=None):
        geom_properties = {
            'group': 1,
            'conaffinity': '0', 
            'contype': '0',
            'density': str(self.density),
            'solref': array_to_string(self.solref),
            'solimp': array_to_string(self.solimp),
        }
        if self.rgba is None and self.geom_rgbas is None:
            # if no color, default to lego material
            geom_properties['material'] = 'lego'
        return self._make_geoms(name=name, site=site, **geom_properties)

    def get_bounding_box_size(self):
        return np.array(self.total_size)

    def in_box(self, position, object_position):
        """
        Checks whether the object is contained within this CompositeObject.
        Useful for when the CompositeObject has holes and the object should
        be within one of the holes. Makes an approximation by treating the
        object as a point, and the CompositeBoxObject as an axis-aligned grid.

        Args:
            position: 3D body position of CompositeObject
            object_position: 3D position of object to test for insertion
        """
        ub = position + self.total_size
        lb = position - self.total_size

        # fudge factor for the z-check, since after insertion the object falls to table
        lb[2] -= 0.01

        return np.all(object_position > lb) and np.all(object_position < ub)


class CompositeBoxObject(CompositeObject):
    """
    An object constructed out of box geoms to make more intricate shapes.
    """

    def __init__(
        self,
        total_size,
        geom_locations,
        geom_sizes,
        geom_names=None,
        geom_rgbas=None,
        geom_frictions=None,
        joint=None,
        rgba=None,
        density=100.,
        solref=[0.02, 1.],
        solimp=[0.9, 0.95, 0.001],
        locations_relative_to_center=False,
        geom_quats=None,
    ):
        super().__init__(
            total_size=total_size,
            geom_types=["box"] * len(geom_locations),
            geom_locations=geom_locations,
            geom_sizes=geom_sizes,
            geom_names=geom_names,
            geom_rgbas=geom_rgbas,
            geom_frictions=geom_frictions,
            joint=joint,
            rgba=rgba,
            density=density,
            solref=solref,
            solimp=solimp,
            locations_relative_to_center=locations_relative_to_center,
            geom_quats=geom_quats,
        )


class BoundingObject(CompositeBoxObject):
    """
    Generates a box with a box-shaped hole cut out of it.
    """

    def __init__(
        self,
        size=[0.1, 0.1, 0.1],
        hole_size=[0.05, 0.05, 0.05],
        hole_location=[0., 0.],
        hole_rgba=None,
        joint=None,
        rgba=None,
        density=100.,
        solref=[0.02, 1.],
        solimp=[0.9, 0.95, 0.001],
        friction=None,
    ):
        """
        NOTE: @hole_location should be relative to the center of the object, and be 2D, since
              the z-location is inferred to be at the top of the box.
        """
        # make sure hole fits within box
        assert np.all(hole_size < size)

        self.hole_size = np.array(hole_size)
        self.hole_rgba = np.array(hole_rgba) if hole_rgba is not None else None
        self.hole_location = np.array(hole_location)

        # specify all geoms in unnormalized position coordinates
        geom_args = self._geoms_from_init(
            size=size, 
            hole_size=self.hole_size, 
            hole_location=self.hole_location, 
            hole_rgba=self.hole_rgba,
            friction=friction,
        )

        super().__init__(
            total_size=size,
            joint=joint, 
            rgba=rgba,
            density=density,
            solref=solref,
            solimp=solimp,
            **geom_args,
        )

    def _geoms_from_init(self, size, hole_size, hole_location, hole_rgba, friction):
        """
        Helper function to retrieve geoms to pass to super class, from the size,
        hole size, and hole location.
        """

        # total size - hole size = remaining space on object
        x_hole_lim = size[0] - hole_size[0]
        y_hole_lim = size[1] - hole_size[1]
        x_hole, y_hole = hole_location[0], hole_location[1]

        # we add a top, bottom, left, and right geom that surround the hole, and
        # a lower base geom that can fill up the bottom of the box to make
        # the hole as shallow as it needs to be.
        geom_names = ['top', 'bottom', 'left', 'right', 'hole_base']
        geom_rgbas = [None, None, None, None, hole_rgba]

        # geom sizes
        #
        # take sizes with hole at center and add sampled hole translation
        top_size = [(x_hole_lim + x_hole) / 2., size[1], size[2]]
        bottom_size = [(x_hole_lim - x_hole) / 2., size[1], size[2]]
        left_size = [size[0], (y_hole_lim + y_hole) / 2., size[2]]
        right_size = [size[0], (y_hole_lim - y_hole) / 2., size[2]]
        hole_base_size = [hole_size[0], hole_size[1], (size[2] - hole_size[2])]
        geom_sizes = [top_size, bottom_size, left_size, right_size, hole_base_size]

        # geom locations
        #
        # top and left are at (0, 0), and bottom and right are just translated by 
        # full size of hole, and top and left respectively
        top_loc = [0, 0, 0]
        bottom_loc = [2. * (top_size[0] + hole_size[0]), 0, 0]
        left_loc = [0, 0, 0]
        right_loc = [0, 2. * (left_size[1] + hole_size[1]), 0]
        hole_base_loc = [2. * top_size[0], 2. * left_size[1], 0]
        geom_locations = [top_loc, bottom_loc, left_loc, right_loc, hole_base_loc]

        # geom frictions
        geom_frictions = [friction for _ in geom_locations]

        return {
            "geom_locations" : geom_locations,
            "geom_sizes" : geom_sizes,
            "geom_names" : geom_names,
            "geom_rgbas" : geom_rgbas,
            "geom_frictions" : geom_frictions,
        }

#     def in_grid(self, position, object_position, object_size):
#         """
#         Args:
#             position: 3D body position of BoundingObject
#             object_position: 3D position of object to test for insertion
#             object_size: 3D array of x, y, and z half-size bounding box dimensions for object
#         """

#         # convert into hole frame
#         rel_pos = np.array(object_position) - np.array(position)

#         # some tolerance for the object size
#         object_size = np.array(object_size) * 0.95

#         # bounds for object and for hole location
#         object_lb = rel_pos - object_size
#         object_ub = rel_pos + object_size
#         hole_lb = self.hole_location - self.hole_size
#         hole_ub = self.hole_location + self.hole_size

#         # fudge factor for the z-check, since after insertion the object falls to table
#         hole_lb[2] -= 0.01
#         return np.all(object_lb > hole_lb) and np.all(object_ub < hole_ub)        


class BoxPatternObject(CompositeBoxObject):
    """
    An object constructed out of box geoms to make more intricate shapes.
    """

    def __init__(
        self,
        unit_size,
        pattern,
        joint=None,
        rgba=None,
        density=100.,
        solref=[0.02, 1.],
        solimp=[0.9, 0.95, 0.001],
        friction=None,
    ):
        """
        Args:
            unit_size (3d array / list): size of each unit block in each dimension

            pattern (3d array / list): array of normalized sizes specifying the
                geometry of the shape. A "0" indicates the absence of a cube and
                a "1" indicates the presence of a full unit block. The dimensions
                correspond to z, x, and y respectively. 
        """

        # number of blocks in z, x, and y
        self.pattern = np.array(pattern)
        self.nz, self.nx, self.ny = self.pattern.shape
        self.unit_size = unit_size

        total_size = [self.nx * unit_size[0], self.ny * unit_size[1], self.nz * unit_size[2]]
        geom_args = self._geoms_from_init(self.unit_size, self.pattern, rgba, friction)
        super().__init__(
            total_size=total_size, 
            joint=joint, 
            rgba=rgba,
            density=density,
            solref=solref,
            solimp=solimp,
            **geom_args,
        )

    def _geoms_from_init(self, unit_size, pattern, rgba, friction):
        """
        Helper function to retrieve geoms to pass to super class.
        """
        geom_locations = []
        geom_sizes = []
        geom_names = []
        nz, nx, ny = pattern.shape
        for k in range(nz):
            for i in range(nx):
                for j in range(ny):
                    if pattern[k, i, j] > 0:
                        geom_sizes.append([
                            unit_size[0], 
                            unit_size[1], 
                            unit_size[2],
                        ])
                        geom_locations.append([
                            i * 2. * unit_size[0], 
                            j * 2. * unit_size[1], 
                            k * 2. * unit_size[2],
                        ])
                        geom_names.append("{}_{}_{}".format(k, i, j))

        geom_rgbas = [rgba for _ in geom_locations]
        geom_frictions = [friction for _ in geom_locations]
        return {
            "geom_locations" : geom_locations,
            "geom_sizes" : geom_sizes,
            "geom_names" : geom_names,
            "geom_rgbas" : geom_rgbas,
            "geom_frictions" : geom_frictions,
        }


class BoundingPatternObject(BoundingObject, BoxPatternObject):
    """
    Generates a box with a box-shaped hole cut out of it.
    The box-shaped hole satisfies a pattern so that more intricate
    voxelized holes are created.
    """

    def __init__(
        self,
        unit_size,
        pattern,
        size=[0.1, 0.1, 0.1],
        hole_location=[0., 0.],
        hole_rgba=None,
        pattern_rgba=None,
        joint=None,
        rgba=None,
        density=100.,
        solref=[0.02, 1.],
        solimp=[0.9, 0.95, 0.001],
        friction=None,
    ):
        """
        NOTE: @hole_location should be relative to the center of the object, and be 2D, since
              the z-location is inferred to be at the top of the box.
        """

        # number of blocks in z, x, and y for the pattern
        self.pattern = np.array(pattern)
        self.pattern_rgba = np.array(pattern_rgba) if pattern_rgba is not None else None
        self.nz, self.nx, self.ny = self.pattern.shape
        self.unit_size = np.array(unit_size)

        self.hole_size = np.array([self.unit_size[0] * self.nx, self.unit_size[1] * self.ny, self.unit_size[2] * self.nz])
        self.hole_rgba = np.array(hole_rgba) if hole_rgba is not None else None
        self.hole_location = np.array(hole_location)

        # make sure hole fits within box
        assert np.all(self.hole_size < size)

        geom_args = self._geoms_from_init(
            unit_size=self.unit_size,
            pattern=self.pattern,
            pattern_rgba=self.pattern_rgba,
            size=size, 
            hole_size=self.hole_size, 
            hole_location=self.hole_location, 
            hole_rgba=self.hole_rgba,
            friction=friction,
        )

        CompositeBoxObject.__init__(
            self,
            total_size=size, 
            joint=joint, 
            rgba=rgba,
            density=density,
            solref=solref,
            solimp=solimp,
            **geom_args,
        )

    def _geoms_from_init(self, unit_size, pattern, pattern_rgba, size, hole_size, hole_location, hole_rgba, friction):
        """
        Helper function to retrieve geoms to pass to super class, from the size,
        hole size, and hole location.
        """
        bounding_geom_args = BoundingObject._geoms_from_init(
            self, 
            size=size, 
            hole_size=hole_size, 
            hole_location=hole_location, 
            hole_rgba=hole_rgba,
            friction=friction,
        )
        pattern_geom_args = BoxPatternObject._geoms_from_init(
            self, 
            unit_size=unit_size,
            pattern=pattern,
            rgba=pattern_rgba,
            friction=friction,
        )

        # use the bottom geom of hole to determine offset for pattern
        hole_base_size = bounding_geom_args["geom_sizes"][-1]
        hole_base_loc = bounding_geom_args["geom_locations"][-1]

        for i in range(len(pattern_geom_args["geom_sizes"])):
            # move locations to account for the bounding box object
            pattern_geom_args["geom_locations"][i][0] += hole_base_loc[0]
            pattern_geom_args["geom_locations"][i][1] += hole_base_loc[1]
            pattern_geom_args["geom_locations"][i][2] += (hole_base_loc[2] + 2. * hole_base_size[2])

        # merge geom lists together
        return {
            "geom_locations" : bounding_geom_args["geom_locations"] + pattern_geom_args["geom_locations"],
            "geom_sizes" : bounding_geom_args["geom_sizes"] + pattern_geom_args["geom_sizes"],
            "geom_names" : bounding_geom_args["geom_names"] + pattern_geom_args["geom_names"],
            "geom_rgbas" : bounding_geom_args["geom_rgbas"] + pattern_geom_args["geom_rgbas"],
            "geom_frictions" : bounding_geom_args["geom_frictions"] + pattern_geom_args["geom_frictions"],
        }


class HollowCylinderObject(CompositeObject):
    """
    Approximates a hollow cylinder with a number of box geoms.
    """
    def __init__(
        self,
        outer_radius=0.0425,
        inner_radius=0.03,
        height=0.05,
        ngeoms=8,
        joint=None,
        rgba=None,
        density=100.,
        make_half=False,
    ):
        # radius of the inner cup hole and entire cup
        self.r1 = inner_radius
        self.r2 = outer_radius

        # number of geoms used to approximate the cylindrical shell
        self.n = ngeoms

        # cylinder half-height
        self.height = height

        # half-width of each box inferred from triangle of radius + box half-length
        # since the angle will be (360 / n) / 2 
        self.unit_box_width = self.r2 * np.sin(np.pi / self.n)

        # half-height of each box inferred from the same triangle with inner radius
        self.unit_box_height = (self.r2 - self.r1) * np.cos(np.pi / self.n) / 2.

        # each box geom depth will end up defining the height of the cup
        self.unit_box_depth = self.height

        # radius of intermediate circle that connects all box centers
        self.int_r = (self.r1 * np.cos(np.pi / self.n)) + self.unit_box_height 

        # if True, will only make half the hollow cylinder
        self.make_half = make_half

        geom_args = self._get_geom_args()

        super().__init__(
            total_size=[self.r2, self.r2, self.height],
            geom_types=geom_args["geom_types"],
            geom_locations=geom_args["geom_locations"],
            geom_sizes=geom_args["geom_sizes"],
            geom_names=None,
            geom_rgbas=None,
            geom_frictions=None,
            joint=joint,
            rgba=rgba,
            density=density,
            solref=[0.02, 1.],
            solimp=[0.998, 0.998, 0.001],
            # solimp=[0.9, 0.95, 0.001],
            locations_relative_to_center=True,
            geom_quats=geom_args["geom_quats"],
        )

    def _get_geom_args(self):

        angle_step = 2. * np.pi / self.n
        box_centers = []
        box_quats = []
        box_sizes = []

        n_make = self.n
        if self.make_half:
            # only make half the shell
            n_make = (self.n // 2) + 1

        for i in range(n_make):
            # we start with the top-most box object and proceed clockwise (thus an offset of np.pi)
            box_angle = np.pi - i * angle_step
            box_centers.append(
                np.array([
                    self.int_r * np.cos(box_angle),
                    self.int_r * np.sin(box_angle),
                    0.
                ])
            )
            box_quats.append(
                np.array([np.cos(box_angle / 2.), 0., 0., np.sin(box_angle / 2.)])
            )
            box_sizes.append(
                np.array([self.unit_box_height, self.unit_box_width, self.unit_box_depth])
            )

        geom_types = ["box"] * len(box_centers)

        return dict(
            geom_types=geom_types,
            geom_locations=box_centers,
            geom_sizes=box_sizes,
            geom_quats=box_quats,
        )


class CupObject(CompositeBodyObject):
    """
    Cup object with optional handle.
    """
    def __init__(
        self,
        outer_cup_radius=0.0425,
        inner_cup_radius=0.03,
        cup_height=0.05,
        cup_ngeoms=8,
        cup_base_height=0.01,
        cup_base_offset=0.005,
        add_handle=False,
        handle_outer_radius=0.03,
        handle_inner_radius=0.015,
        handle_thickness=0.005,
        handle_ngeoms=8,
        joint=None,
        rgba=None,
        density=100.,
    ):

        # radius of the inner cup hole and entire cup
        self.r1 = inner_cup_radius
        self.r2 = outer_cup_radius

        # number of geoms used to approximate the cylindrical shell
        self.n = cup_ngeoms

        # cup half-height
        self.cup_height = cup_height

        # cup base args
        self.cup_base_height = cup_base_height
        self.cup_base_offset = cup_base_offset

        # handle args
        self.add_handle = add_handle
        self.handle_outer_radius = handle_outer_radius
        self.handle_inner_radius = handle_inner_radius
        self.handle_thickness = handle_thickness
        self.handle_ngeoms = handle_ngeoms

        objects = []
        object_locations = []
        object_quats = []

        # cup body
        self.cup_body = HollowCylinderObject(
            outer_radius=self.r2,
            inner_radius=self.r1,
            height=self.cup_height,
            ngeoms=self.n,
            rgba=rgba,
            density=density,
            joint=[],
        )
        objects.append(self.cup_body)
        object_locations.append([0., 0., 0.])
        object_quats.append([1., 0., 0., 0.])

        # cup base
        self.cup_base = CylinderObject(
            size=[self.cup_body.int_r, self.cup_base_height],
            rgba=rgba,
            density=density,
            solref=[0.02, 1.],
            solimp=[0.998, 0.998, 0.001],
            joint=[],
        )
        objects.append(self.cup_base)
        object_locations.append([0., 0., -self.cup_height + self.cup_base_height + self.cup_base_offset])
        object_quats.append([1., 0., 0., 0.])

        if self.add_handle:
            # cup handle is a hollow half-cylinder
            self.cup_handle = HollowCylinderObject(
                outer_radius=self.handle_outer_radius,
                inner_radius=self.handle_inner_radius,
                height=self.handle_thickness,
                ngeoms=self.handle_ngeoms,
                rgba=rgba,
                density=density,
                joint=[],
                make_half=True,
            )
            # translate handle to right side of cup body, and rotate by +90 degrees about y-axis 
            # to orient the handle geoms on the cup body
            objects.append(self.cup_handle)
            object_locations.append([0., (self.cup_body.r2 + self.cup_handle.unit_box_width), 0.])
            object_quats.append(
                T.convert_quat(
                    T.mat2quat(T.rotation_matrix(angle=np.pi / 2., direction=[0., 1., 0.])[:3, :3]),
                    to="wxyz",
                )
            )

        body_total_size = [self.r2, self.r2, self.cup_height]
        if self.add_handle:
            body_total_size[1] += self.handle_outer_radius

        super().__init__(
            objects=objects,
            total_size=body_total_size,
            object_locations=object_locations,
            joint=joint,
            locations_relative_to_center=True,
            object_quats=object_quats,
        )


class BoxObject(MujocoGeneratedObject):
    """
    An object that is a box
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
        horizontal_radius_offset=0.,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07, 0.07],
                         [0.03, 0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        self.horizontal_radius_offset = horizontal_radius_offset
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 3, "box size should have length 3"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[2]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[2]])

    def get_horizontal_radius(self):
        return np.linalg.norm(self.size[0:2], 2) + self.horizontal_radius_offset

    def get_bounding_box_size(self):
        return np.array([self.size[0], self.size[1], self.size[2]])

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="box")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="box")


class CylinderObject(MujocoGeneratedObject):
    """
    A randomized cylinder object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "cylinder size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[1]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[1]])

    def get_horizontal_radius(self):
        return self.size[0]

    def get_bounding_box_size(self):
        return np.array([self.size[0], self.size[0], self.size[1]])

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="cylinder")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="cylinder")


class BallObject(MujocoGeneratedObject):
    """
    A randomized ball (sphere) object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07],
                         [0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 1, "ball size should have length 1"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * self.size[0]])

    def get_top_offset(self):
        return np.array([0, 0, self.size[0]])

    def get_horizontal_radius(self):
        return self.size[0]

    def get_bounding_box_size(self):
        return np.array([self.size[0], self.size[0], self.size[0]])

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="sphere")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="sphere")


class CapsuleObject(MujocoGeneratedObject):
    """
    A randomized capsule object.
    """

    def __init__(
        self,
        size=None,
        size_max=None,
        size_min=None,
        density=None,
        density_range=None,
        friction=None,
        friction_range=None,
        rgba="random",
        joint=None,
        solref=None,
        solimp=None,
    ):
        size = _get_size(size,
                         size_max,
                         size_min,
                         [0.07, 0.07],
                         [0.03, 0.03])
        density_range = _get_randomized_range(density,
                                              density_range,
                                              DEFAULT_DENSITY_RANGE)
        friction_range = _get_randomized_range(friction,
                                               friction_range,
                                               DEFAULT_FRICTION_RANGE)
        super().__init__(
            size=size,
            rgba=rgba,
            density_range=density_range,
            friction=friction,
            friction_range=friction_range,
            joint=joint,
            solref=solref,
            solimp=solimp,
        )

    def sanity_check(self):
        assert len(self.size) == 2, "capsule size should have length 2"

    def get_bottom_offset(self):
        return np.array([0, 0, -1 * (self.size[0] + self.size[1])])

    def get_top_offset(self):
        return np.array([0, 0, (self.size[0] + self.size[1])])

    def get_horizontal_radius(self):
        return self.size[0]

    def get_bounding_box_size(self):
        return np.array([self.size[0], self.size[0], self.size[0] + self.size[1]])

    # returns a copy, Returns xml body node
    def get_collision(self, name=None, site=False):
        return self._get_collision(name=name, site=site, ob_type="capsule")

    # returns a copy, Returns xml body node
    def get_visual(self, name=None, site=False):
        return self._get_visual(name=name, site=site, ob_type="capsule")


### More Miscellaneous Objects ###


class AnimalObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.033)
        self.body_y = random.uniform(0.015,0.03)
        self.body_z = random.uniform(0.01,0.035)
        self.legs_x = random.uniform(0.005,0.01)
        self.legs_z = random.uniform(0.01,0.035)
        self.neck_x = random.uniform(0.005,0.01)
        self.neck_z = random.uniform(0.005,0.01)
        self.head_y = random.uniform(0.010,0.015)
        self.head_z = random.uniform(0.005,0.01)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.legs_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.neck_z+2*self.head_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+self.body_y**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #legs
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[0.9*self.body_x-self.legs_x, 0.9*self.body_y-self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[-0.9*self.body_x+self.legs_x, 0.9*self.body_y-self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[0.9*self.body_x-self.legs_x, -0.9*self.body_y+self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.legs_x,self.legs_z],pos=[-0.9*self.body_x+self.legs_x, -0.9*self.body_y+self.legs_x, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #neck
        main_body.append(
        new_geom(
            geom_type="box", size=[self.neck_x,self.neck_x,self.neck_z],pos=[self.body_x-self.neck_x, 0, self.neck_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="box", size=[self.head_y,self.neck_x*1.5,self.head_z],pos=[self.body_x-2*self.neck_x+self.head_y, 0, 2*self.neck_z+self.body_z+self.head_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class CarObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.031)
        self.body_y = random.uniform(0.015,0.03)
        self.body_z = random.uniform(0.01,self.body_x/2)
        self.wheels_r = random.uniform(self.body_x/4.0,self.body_x/3.0)
        self.wheels_z = random.uniform(0.002,0.004)
        self.top_x = random.uniform(0.008,0.9*self.body_x)
        self.top_y = random.uniform(0.007,0.9*self.body_y)
        self.top_z = random.uniform(0.004,0.9*self.body_z)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-self.wheels_r])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.top_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+(self.body_y+2*self.wheels_z)**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #wheels
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #top
        main_body.append(
        new_geom(
            geom_type="box", size=[self.top_x,self.top_y,self.top_z],pos=[0, 0, self.top_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class TrainObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.017,0.031)
        self.body_y = random.uniform(0.025,0.045)
        self.body_z = random.uniform(0.01,0.025)
        self.wheels_r = random.uniform(self.body_x/4.0,self.body_x/3.0)
        self.wheels_z = random.uniform(0.002,0.006)
        self.top_x = random.uniform(0.01,0.9*self.body_x)
        self.top_r = 0.99*self.body_x
        self.top_z = 0.99*self.body_y
        self.cabin_x = 0.99*self.body_x
        self.cabin_y = random.uniform(0.20,0.3)*self.body_y
        self.cabin_z = random.uniform(0.5,0.8)*self.top_r
        self.chimney_r = random.uniform(0.004,0.01)
        self.chimney_z = random.uniform(0.01,0.03)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-self.wheels_r])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.chimney_z+self.top_r])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+(self.body_y+2*self.wheels_z)**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #wheels
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, self.body_y-self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.wheels_r,self.wheels_z],pos=[-self.body_x, -self.body_y+self.wheels_r, -self.body_z], group=1, zaxis='1 0 0',
             rgba=np.append(np.random.uniform(size=3),1),)
        )
        #top
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.top_r,self.top_z],pos=[0, 0, self.body_z], group=1, zaxis="0 1 0",
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #cabin
        main_body.append(
        new_geom(
            geom_type="box", size=[self.cabin_x,self.cabin_y,self.cabin_z],pos=[0, -self.body_y+self.cabin_y, self.body_z+self.cabin_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #chimney
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.chimney_r,self.chimney_z],pos=[0, self.body_y*.5, self.body_z+self.top_r], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class BipedObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_x = random.uniform(0.02,0.031)
        self.body_y = random.uniform(0.017,0.022)
        self.body_z = random.uniform(0.015,0.03)
        self.legs_x = random.uniform(0.005,0.01)
        self.legs_z = random.uniform(0.005,self.body_z)
        self.hands_x = random.uniform(0.005,0.01)
        self.hands_z = random.uniform(0.01,0.3*self.legs_z)
        self.head_y = self.body_y
        self.head_z = random.uniform(0.01,0.02)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.legs_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return np.sqrt(self.body_x**2+self.body_y**2)

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="box", size=[self.body_x,self.body_y,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #legs
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.body_y,self.legs_z],pos=[self.body_x-self.legs_x, 0, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.legs_x,self.body_y,self.legs_z],pos=[-self.body_x+self.legs_x, 0, -self.legs_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )

        #hands
        main_body.append(
        new_geom(
            geom_type="box", size=[self.hands_x,2*self.body_y,self.hands_z],pos=[self.body_x+self.hands_x, self.body_y, -self.hands_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="box", size=[self.hands_x,2*self.body_y,self.hands_z],pos=[-self.body_x-self.hands_x, self.body_y, -self.hands_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="box", size=[self.head_y,self.head_y,self.head_z],pos=[0, 0, self.body_z+self.head_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class DumbbellObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.009,0.013)
        self.body_z = random.uniform(0.015,0.025)
        self.head_r = random.uniform(1.6*self.body_r,2*self.body_r)
        self.head_z = random.uniform(0.005,0.01)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z-2*self.head_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_z+self.head_z

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, -self.head_z-self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, self.head_z+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )        

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)

class HammerObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.009,0.013)
        self.body_z = random.uniform(0.027,0.037)
        self.head_r = random.uniform(1.6*self.body_r,3*self.body_r)
        self.head_z = random.uniform(1.5*self.body_r,2*self.body_r)
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_r+self.head_r

    def get_collision(self, name=None, site=None):
        main_body = new_body()

        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, 0, 0], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, 0.95*self.head_r+self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1),zaxis='1 0 0')
        )
    

        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


class GuitarObject(MujocoGeneratedObject):
    """
    Generates bounding box hole object
    """

    def __init__(self):
        super().__init__()
        # generate random vector
        self.body_r = random.uniform(0.021,0.027)/1.7
        self.body_z = random.uniform(0.017,0.025)/1.4
        self.head_r = random.uniform(1.5,2)*self.body_r
        self.head_z = self.body_z
        self.arm_x = random.uniform(0.008,0.010)/2
        self.arm_y = random.uniform(1.2,1.6)*(self.body_r+self.head_r)
        self.arm_z = 0.007/2
    def get_bottom_offset(self):
        return np.array([0, 0, -self.body_z])

    def get_top_offset(self):
        return np.array([0, 0, self.body_z+2*self.head_z])

    def get_horizontal_radius(self):
        return self.body_r+self.head_r

    def get_collision(self, name=None, site=None):
        main_body = new_body()
        color = np.append(np.random.uniform(size=3),1)
        if name is not None:
            main_body.set("name", name)
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.body_r,self.body_z],pos=[0, self.head_r+0.5*self.body_r, 0], group=1,
             rgba=color)
        )
        #head
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r,self.head_z],pos=[0, 0, 0], group=1,
             rgba=color)
        )
        main_body.append(
        new_geom(
            geom_type="cylinder", size=[self.head_r*0.5,self.head_z],pos=[0, 0, 0.001], group=1,
             rgba=[0,0,0,1])
        )
        #arm
        main_body.append(
        new_geom(
            geom_type="box", size=[self.arm_x,self.arm_y,self.arm_z],pos=[0, self.arm_y, self.body_z], group=1,
             rgba=np.append(np.random.uniform(size=3),1))
        )
        if site:
            # add a site as well
            template = self.get_site_attrib_template()
            if name is not None:
                template["name"] = name
            main_body.append(ET.Element("site", attrib=template))
        return main_body

    def get_visual(self, name=None, site=None):
        return self.get_collision(name, site)


