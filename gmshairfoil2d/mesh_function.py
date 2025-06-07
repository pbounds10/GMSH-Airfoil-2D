import math
import os
import sys
from pathlib import Path

import gmsh
import numpy as np

from airfoil_func import NACA_4_digit_geom, get_airfoil_points
from geometry_def import (AirfoilSpline, Circle, CType,
                                        PlaneSurface, Rectangle, outofbounds)


def mesh_airfoil(
    output_path: str,
    output_name: str, 
    naca: int = None,
    airfoil: str = None,
    local_airfoil: str = None,
    aoa: float = 0.0,
    airfoil_mesh_size: float = 0.01, 
    structural: bool = False,
    arg_struc: str = None, #"10x10"
    ratio: float = 1.2,
    no_bl: bool = False,
    first_layer: float = 3e-5,
    nb_layers: int = 10,
    box: str = None, #"10x10"
    farfield: float = 10,
    ext_mesh_size: float = 0.2,
    output_format: str = "msh",
    wake_size_start: float = 0.01,
    wake_size_end: float = 1.2,
):

    # Airfoil choice
    cloud_points = None
    if naca:
        airfoil_name = naca
        cloud_points = NACA_4_digit_geom(airfoil_name)

    if airfoil:
        airfoil_name = airfoil
        cloud_points = get_airfoil_points(airfoil_name)

    if local_airfoil:
        airfoil_path = local_airfoil
        airfoil_name = os.path.basename(airfoil_path).split(".")[0]
        cloud_points = get_airfoil_points(airfoil_path, local=True)

    if cloud_points is None:
        print("\nNo airfoil profile specified, exiting")
        print("You must use --naca or --airfoil\n")
        sys.exit()

    # Make the points all start by the (0,0) (or minimum of coord x when not exactly 0) and go clockwise
    # --> to be easier to deal with after (in airfoilspline)
    le = min(p[0] for p in cloud_points)
    for p in cloud_points:
        if p[0] == le:
            debut = cloud_points.index(p)
    cloud_points = cloud_points[debut:]+cloud_points[:debut]
    if cloud_points[1][1] < cloud_points[0][1]:
        cloud_points.reverse()
        cloud_points = cloud_points[-1:] + cloud_points[:-1]

    # Angle of attack
    aoa = -aoa * (math.pi / 180)

    # Generate Geometry
    gmsh.initialize()

    # Airfoil
    airfoil = AirfoilSpline(
        cloud_points, airfoil_mesh_size)
    airfoil.rotation(aoa, (0.5, 0, 0), (0, 0, 1))
    gmsh.model.geo.synchronize()

    # If structural, all is done in CType
    if structural:
        dx_wake, dy = [float(value)for value in arg_struc.split("x")]
        mesh = CType(airfoil, dx_wake, dy,
                     airfoil_mesh_size, first_layer, ratio, aoa)
        mesh.define_bc()

    else:
        k1, k2 = airfoil.gen_skin()
        # Choose the parameters for bl (when exist)
        if not no_bl:
            N = nb_layers
            r = ratio
            d = [first_layer]
            # Construct the vector of cumulative distance of each layer from airfoil
            for i in range(1, N):
                d.append(d[-1] - (-d[0]) * r**i)
        else:
            d = [0]

        # Need to check that the layers or airfoil do not go outside the box/circle (d[-1] is the total height of bl)
        outofbounds(airfoil, box, farfield, d[-1])

        # External domain
        if box:
            length, width = [float(value) for value in box.split("x")]
            ext_domain = Rectangle(0.5, 0, 0, length, width,
                                   mesh_size=ext_mesh_size)
        else:
            ext_domain = Circle(0.5, 0, 0, radius=farfield,
                                mesh_size=ext_mesh_size)
        gmsh.model.geo.synchronize()

        # Create the surface for the mesh
        surface = PlaneSurface([ext_domain, airfoil])
        gmsh.model.geo.synchronize()

        # Create the boundary layer
        if not no_bl:
            curv = [airfoil.upper_spline.tag,
                    airfoil.lower_spline.tag, airfoil.front_spline.tag]

            # Creates a new mesh field of type 'BoundaryLayer' and assigns it an ID (f).
            f = gmsh.model.mesh.field.add('BoundaryLayer')

            # Add the curves where we apply the boundary layer (around the airfoil for us)
            gmsh.model.mesh.field.setNumbers(f, 'CurvesList', curv)
            gmsh.model.mesh.field.setNumber(f, 'Size', d[0])  # size 1st layer
            gmsh.model.mesh.field.setNumber(f, 'Ratio', r)  # Growth ratio
            # Total thickness of boundary layer
            gmsh.model.mesh.field.setNumber(f, 'Thickness', d[-1])

            # Forces to use quads and not triangle when =1 (i.e. true)
            gmsh.model.mesh.field.setNumber(f, 'Quads', 1)

            # Enter the points where we want a "fan" (points must be at end on line)(only te for us)
            gmsh.model.mesh.field.setNumbers(
                f, "FanPointsList", [airfoil.te.tag])

            gmsh.model.mesh.field.setAsBoundaryLayer(f)

        # Define boundary conditions (name the curves)
        # ext_domain.define_bc()
        # surface.define_bc()
        # airfoil.define_bc()

    gmsh.model.geo.synchronize()

    # Choose the parameters of the mesh : we want the mesh size according to the points and not curvature (doesn't work with farfield)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    if not structural and not no_bl:
        # Add transfinite line on the front to get more point in the middle (where the curvature of the le makes it usually more spaced)
        x, y, v, w = airfoil.points[k1].x, airfoil.points[k2].y, airfoil.points[k1].x, airfoil.points[k2].y
        c1, c2 = airfoil.le.x, airfoil.le.y
        # To get an indication of numbers of points needed, compute approximate length of curve of front spline
        l = (math.sqrt((x-c1)*(x-c1)+(y-c2)*(y-c2)) +
             math.sqrt((v-c1)*(v-c1)+(w-c2)*(w-c2)))
        # As points will be more near than mesh size on the front, need more points
        nb_points = int(3.5*l/airfoil_mesh_size)
        gmsh.model.mesh.setTransfiniteCurve(
            airfoil.front_spline.tag, nb_points, "Bump", 10)
        # Choose the nbs of points in the fan at the te:
        # Compute coef : between 10 and 25, 15 when usual mesh size but adapted to mesh size
        coef = max(10, min(25, 15*0.01/airfoil_mesh_size))
        gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", coef)

    #creat a size field to control the wake
    print(farfield)
    gmsh.model.mesh.field.add("Frustum", 2)
    gmsh.model.mesh.field.setNumber(2, "InnerR1", 0.001 )
    gmsh.model.mesh.field.setNumber(2, "InnerR2", 0.001 )
    gmsh.model.mesh.field.setNumber(2, "InnerV1", wake_size_start )
    gmsh.model.mesh.field.setNumber(2, "OuterV1", wake_size_start )
    gmsh.model.mesh.field.setNumber(2, "OuterR1", 0.2 )
    gmsh.model.mesh.field.setNumber(2, "OuterR2", 5 )
    gmsh.model.mesh.field.setNumber(2, "InnerV2", wake_size_end )
    gmsh.model.mesh.field.setNumber(2, "OuterV2", wake_size_end )
    gmsh.model.mesh.field.setNumber(2, "X1", -0.1 )
    gmsh.model.mesh.field.setNumber(2, "X2", farfield*1.25 )
    gmsh.model.mesh.field.setNumber(2, "Y1", 0 )
    gmsh.model.mesh.field.setNumber(2, "Y2", 0 )
    gmsh.model.mesh.field.setNumber(2, "Z1", 0 )
    gmsh.model.mesh.field.setNumber(2, "Z2", 0 )
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    # Generate mesh
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D", 5)
    h = 1
    extrude_surface = gmsh.model.geo.extrude([(2, 1)], 0, 0, h, numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, tags=[1], name="back")
    gmsh.model.addPhysicalGroup(2, tags=[extrude_surface[0][1]], name="front")
    farfield_surface_indices = []
    for surf_index in extrude_surface[2:-3]:
        farfield_surface_indices.append(surf_index[1])
    gmsh.model.addPhysicalGroup(2, tags=farfield_surface_indices, name=f"farfield")

    airfoil_surface_indices = []
    for surf_index in extrude_surface[-3:]:
        airfoil_surface_indices.append(surf_index[1])
    gmsh.model.addPhysicalGroup(2, tags=airfoil_surface_indices, name=f"airfoil")

    gmsh.model.addPhysicalGroup(3, tags=[extrude_surface[1][1]], tag=10001, name="fluid_volume")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)
    # gmsh.model.mesh.optimize()

    # Mesh file name and output
    mesh_path = Path(
        output_path, f"{output_name}.{output_format}")
    gmsh.write(str(mesh_path))
    gmsh.finalize()

def create_parent_mesh(
    output_path: str,
    output_name: str, 
    structural: bool = False,
    arg_struc: str = None, #"10x10"
    box: str = None, #"10x10"
    farfield: float = 10,
    ext_mesh_size: float = 0.2,
    output_format: str = "msh",
    wake_size_start: float = 0.01,
    wake_size_end: float = 1.2,
):

    # Airfoil choice

    # if cloud_points is None:
    #     print("\nNo airfoil profile specified, exiting")
    #     print("You must use --naca or --airfoil\n")
    #     sys.exit()

    # Make the points all start by the (0,0) (or minimum of coord x when not exactly 0) and go clockwise
    # --> to be easier to deal with after (in airfoilspline)
    # le = min(p[0] for p in cloud_points)
    # for p in cloud_points:
    #     if p[0] == le:
    #         debut = cloud_points.index(p)
    # cloud_points = cloud_points[debut:]+cloud_points[:debut]
    # if cloud_points[1][1] < cloud_points[0][1]:
    #     cloud_points.reverse()
    #     cloud_points = cloud_points[-1:] + cloud_points[:-1]

    # Angle of attack
    # aoa = -aoa * (math.pi / 180)

    # Generate Geometry
    gmsh.initialize()

    # Airfoil
    # airfoil = AirfoilSpline(
    #     cloud_points, airfoil_mesh_size)
    # airfoil.rotation(aoa, (0.5, 0, 0), (0, 0, 1))
    gmsh.model.geo.synchronize()

    # If structural, all is done in CType
    # if structural:
    #     dx_wake, dy = [float(value)for value in arg_struc.split("x")]
    #     mesh = CType(airfoil, dx_wake, dy,
    #                  airfoil_mesh_size, first_layer, ratio, aoa)
    #     mesh.define_bc()

    if True:
        # k1, k2 = airfoil.gen_skin()
        # # Choose the parameters for bl (when exist)
        # if not no_bl:
        #     N = nb_layers
        #     r = ratio
        #     d = [first_layer]
        #     # Construct the vector of cumulative distance of each layer from airfoil
        #     for i in range(1, N):
        #         d.append(d[-1] - (-d[0]) * r**i)
        # else:
        #     d = [0]

        # Need to check that the layers or airfoil do not go outside the box/circle (d[-1] is the total height of bl)
        # outofbounds(None, box, farfield, d[-1])

        # External domain
        if box:
            length, width = [float(value) for value in box.split("x")]
            ext_domain = Rectangle(0.5, 0, 0, length, width,
                                   mesh_size=ext_mesh_size)
        else:
            ext_domain = Circle(0.5, 0, 0, radius=farfield,
                                mesh_size=ext_mesh_size)
        gmsh.model.geo.synchronize()

        # Create the surface for the mesh
        surface = PlaneSurface([ext_domain])
        gmsh.model.geo.synchronize()

        # Create the boundary layer
        # if not no_bl:
        #     curv = [airfoil.upper_spline.tag,
        #             airfoil.lower_spline.tag, airfoil.front_spline.tag]

        #     # Creates a new mesh field of type 'BoundaryLayer' and assigns it an ID (f).
        #     f = gmsh.model.mesh.field.add('BoundaryLayer')

        #     # Add the curves where we apply the boundary layer (around the airfoil for us)
        #     gmsh.model.mesh.field.setNumbers(f, 'CurvesList', curv)
        #     gmsh.model.mesh.field.setNumber(f, 'Size', d[0])  # size 1st layer
        #     gmsh.model.mesh.field.setNumber(f, 'Ratio', r)  # Growth ratio
        #     # Total thickness of boundary layer
        #     gmsh.model.mesh.field.setNumber(f, 'Thickness', d[-1])

        #     # Forces to use quads and not triangle when =1 (i.e. true)
        #     gmsh.model.mesh.field.setNumber(f, 'Quads', 1)

        #     # Enter the points where we want a "fan" (points must be at end on line)(only te for us)
        #     gmsh.model.mesh.field.setNumbers(
        #         f, "FanPointsList", [airfoil.te.tag])

        #     gmsh.model.mesh.field.setAsBoundaryLayer(f)

        # Define boundary conditions (name the curves)
        # ext_domain.define_bc()
        # surface.define_bc()
        # airfoil.define_bc()

    gmsh.model.geo.synchronize()

    # # Choose the parameters of the mesh : we want the mesh size according to the points and not curvature (doesn't work with farfield)
    # gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 1)
    # gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    # if not structural and not no_bl:
    #     # Add transfinite line on the front to get more point in the middle (where the curvature of the le makes it usually more spaced)
    #     x, y, v, w = airfoil.points[k1].x, airfoil.points[k2].y, airfoil.points[k1].x, airfoil.points[k2].y
    #     c1, c2 = airfoil.le.x, airfoil.le.y
    #     # To get an indication of numbers of points needed, compute approximate length of curve of front spline
    #     l = (math.sqrt((x-c1)*(x-c1)+(y-c2)*(y-c2)) +
    #          math.sqrt((v-c1)*(v-c1)+(w-c2)*(w-c2)))
    #     # As points will be more near than mesh size on the front, need more points
    #     nb_points = int(3.5*l/airfoil_mesh_size)
    #     gmsh.model.mesh.setTransfiniteCurve(
    #         airfoil.front_spline.tag, nb_points, "Bump", 10)
    #     # Choose the nbs of points in the fan at the te:
    #     # Compute coef : between 10 and 25, 15 when usual mesh size but adapted to mesh size
    #     coef = max(10, min(25, 15*0.01/airfoil_mesh_size))
    #     gmsh.option.setNumber("Mesh.BoundaryLayerFanElements", coef)

    #creat a size field to control the wake
    print(farfield)
    gmsh.model.mesh.field.add("Frustum", 2)
    gmsh.model.mesh.field.setNumber(2, "InnerR1", 0.001 )
    gmsh.model.mesh.field.setNumber(2, "InnerR2", 0.001 )
    gmsh.model.mesh.field.setNumber(2, "InnerV1", wake_size_start )
    gmsh.model.mesh.field.setNumber(2, "OuterV1", wake_size_start )
    gmsh.model.mesh.field.setNumber(2, "OuterR1", 0.2 )
    gmsh.model.mesh.field.setNumber(2, "OuterR2", 5 )
    gmsh.model.mesh.field.setNumber(2, "InnerV2", wake_size_end )
    gmsh.model.mesh.field.setNumber(2, "OuterV2", wake_size_end )
    gmsh.model.mesh.field.setNumber(2, "X1", -0.1 )
    gmsh.model.mesh.field.setNumber(2, "X2", farfield*1.25 )
    gmsh.model.mesh.field.setNumber(2, "Y1", 0 )
    gmsh.model.mesh.field.setNumber(2, "Y2", 0 )
    gmsh.model.mesh.field.setNumber(2, "Z1", 0 )
    gmsh.model.mesh.field.setNumber(2, "Z2", 0 )

    gmsh.model.mesh.field.add("Cylinder", 3)
    gmsh.model.mesh.field.setNumber(3, "Radius", 1 )
    gmsh.model.mesh.field.setNumber(3, "VIn", wake_size_start )
    gmsh.model.mesh.field.setNumber(3, "VOut", ext_mesh_size )
    gmsh.model.mesh.field.setNumber(3, "XAxis", 0 )
    gmsh.model.mesh.field.setNumber(3, "XCenter", 0 )
    gmsh.model.mesh.field.setNumber(3, "YAxis", 0 )
    gmsh.model.mesh.field.setNumber(3, "YCenter", 0 )
    gmsh.model.mesh.field.setNumber(3, "ZAxis", 1 )
    gmsh.model.mesh.field.setNumber(3, "ZCenter", 0 )

    gmsh.model.mesh.field.add("Min", 4)
    gmsh.model.mesh.field.setNumbers(4, "FieldsList", [2,3] )

    gmsh.model.mesh.field.setAsBackgroundMesh(4)

    # Generate mesh
    gmsh.model.geo.synchronize()
    gmsh.option.setNumber("Mesh.Algorithm", 6)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D", 5)
    h = 1
    extrude_surface = gmsh.model.geo.extrude([(2, 1)], 0, 0, h, numElements=[1], recombine=True)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, tags=[1], name="back")
    gmsh.model.addPhysicalGroup(2, tags=[extrude_surface[0][1]], name="front")
    farfield_surface_indices = []
    for surf_index in extrude_surface[2:-3]:
        farfield_surface_indices.append(surf_index[1])
    gmsh.model.addPhysicalGroup(2, tags=farfield_surface_indices, name=f"farfield")

    airfoil_surface_indices = []
    for surf_index in extrude_surface[-3:]:
        airfoil_surface_indices.append(surf_index[1])
    gmsh.model.addPhysicalGroup(2, tags=airfoil_surface_indices, name=f"airfoil")

    gmsh.model.addPhysicalGroup(3, tags=[extrude_surface[1][1]], tag=10001, name="fluid_volume")
    gmsh.model.geo.synchronize()

    gmsh.model.mesh.generate(3)
    # gmsh.model.mesh.optimize()

    # Mesh file name and output
    mesh_path = Path(
        output_path, f"{output_name}.{output_format}")
    gmsh.write(str(mesh_path))
    gmsh.finalize()
