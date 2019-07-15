# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:04:48 2016

@author: Martin Friedl
"""

import numpy as np

from gdsCAD_py3.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_py3.shapes import Box, Rectangle, Label, LineLabel, Disk, RegPolygon
from gdsCAD_py3.templates111 import dashed_line

# Layers
l_smBeam = 0
l_lgBeam = 1

smFrameSize = 400.


class Frame(Cell):
    """
    Make a frame for writing to with ebeam lithography
    Params:
    -name of the frame, just like when naming a cell
    -size: the size of the frame as an array [xsize,ysize]
    """

    def __init__(self, name, size, borderLayers):
        if not (type(borderLayers) == list): borderLayers = [borderLayers]
        Cell.__init__(self, name)
        self.size_x, self.size_y = size
        # Create the border of the cell
        for l in borderLayers:
            self.border = Box(
                (-self.size_x / 2., -self.size_y / 2.),
                (self.size_x / 2., self.size_y / 2.),
                1,
                layer=l)
            self.add(self.border)  # Add border to the frame

    def makeAlignMarkers(self, t, w, position, layers, cross=False):
        if not (type(layers) == list): layers = [layers]
        self.aMarkers = Cell("AlignMarkers")
        for l in layers:
            if not (cross):
                am1 = Rectangle((-w / 2., -w / 2.), (w / 2., w / 2.), layer=l)
            elif cross:
                h = w
                crosspts = [(-t / 2., t / 2.), (-t / 2., h / 2.), (t / 2., h / 2.),
                            (t / 2., t / 2.), (w / 2., t / 2.), (w / 2., -t / 2.),
                            (t / 2., -t / 2.), (t / 2., -h / 2.),
                            (-t / 2., -h / 2.), (-t / 2., -t / 2.),
                            (-w / 2., -t / 2.), (-w / 2., t / 2.)]
                am1 = Boundary(crosspts, layer=l)  # Create gdsCAD shape
                # am1 = Polygon(crosspts) #Create shapely polygon for later calculation
            am1 = am1.translate(tuple(position))  # 850,850
            am2 = am1.copy().scale((-1, 1))  # Reflect in x-axis
            am3 = am1.copy().scale((1, -1))  # Reflect in y-axis
            am4 = am1.copy().scale((-1, -1))  # Reflect in both x and y-axis
            self.aMarkers.add([am1, am2, am3, am4])
            self.add(self.aMarkers)

    def processCheck_CleavingMark(self, yPosition, layers):
        if not (type(layers) == list): layers = [layers]
        for l in layers:
            pt1 = (100, yPosition)
            pt2 = (2000, yPosition)
            pt3 = (-1100, yPosition)
            pt4 = (-2800, yPosition)
            line1 = Path([pt1, pt2], width=10, layer=l)
            line2 = Path([pt3, pt4], width=10, layer=l)
            cMarkCell = Cell('Cleaving Mark')
            cMarkCell.add(line1)
            cMarkCell.add(line2)
            self.add(cMarkCell)

    def processCheck_Slits(self, position, arrayWidth, slitWidth, pitch, length, rotation, layers):
        if not (type(layers) == list): layers = [layers]
        Nx = int(arrayWidth / pitch)
        Ny = 1
        for l in layers:
            # Define the slits
            slit = Cell("Slits")
            rect = Rectangle(
                (-slitWidth / 2., -length / 2.),
                (slitWidth / 2., length / 2.),
                layer=l)
            slit.add(rect)
            slits = CellArray(slit, Nx, Ny, (pitch, 0))
            slits.translate((-(Nx) * (pitch) / 2., 0.))
            slits.translate(position)
            slitarray = Cell("ProcessCheckingSlits")
            slitarray.add(slits)
        self.add(slitarray)

    def makeYShapes(self, length, width, rotAngle, spacing, Nx, Ny, layers):
        if not (type(layers) == list): layers = [layers]
        slit = Cell("Slit")
        for l in layers:
            membrane = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=l)
            membrane_cell = Cell('Membrane_w{:.0f}'.format(width * 1000))
            membrane_cell.add(membrane)
            slit.add(membrane_cell)
        shape = Cell('Shapes')
        shape.add(slit, rotation=0 + rotAngle)
        shape.add(slit, rotation=120 + rotAngle)
        shape.add(slit, rotation=240 + rotAngle)

        #            CellArray(slit, Nx, Ny,(length + spacing, pitchV))
        xspacing = length + spacing
        yspacing = (length + spacing) * np.sin(np.deg2rad(60))
        shapearray = CellArray(shape, Nx, Ny / 2, (xspacing, yspacing * 2.), origin=(
            -(Nx * xspacing - spacing) / 2., -(Ny * yspacing - spacing * np.sin(np.deg2rad(60))) / 2.))
        shapearray2 = CellArray(shape, Nx, Ny / 2, (xspacing, yspacing * 2.), origin=(
            xspacing / 2. - (Nx * xspacing - spacing) / 2.,
            yspacing - (Ny * yspacing - spacing * np.sin(np.deg2rad(60))) / 2.))

        allshapes = Cell('All Shapes')
        allshapes.add(shapearray)
        allshapes.add(shapearray2)
        self.add(allshapes)

    def makeTriShapes(self, length, width, rotAngle, spacing, Nx, Ny, layers):
        if not (type(layers) == list): layers = [layers]
        # pt1 = np.array((-length / 2. - width / 4., -width / 2.)) + np.array([0, np.tan(np.deg2rad(30)) * length / 2.])
        # pt2 = np.array((length / 2. + width / 4., width / 2.)) + np.array([0, np.tan(np.deg2rad(30)) * length / 2.])
        slit = Cell("Slit")
        for l in layers:
            membrane = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=l)
            membrane_cell = Cell('Membrane_w{:.0f}'.format(width * 1000))
            membrane_cell.add(membrane)
            slit.add(membrane_cell)
        shape = Cell('Shapes')
        shape.add(slit, rotation=0 + rotAngle)
        shape.add(slit, rotation=120 + rotAngle)
        shape.add(slit, rotation=240 + rotAngle)

        #            CellArray(slit, Nx, Ny,(length + spacing, pitchV))
        xspacing = length + spacing
        yspacing = (length + spacing) * np.tan(np.deg2rad(60)) / 2.
        shapearray = CellArray(shape, Nx, Ny / 2, (xspacing, yspacing * 2.), origin=(
            -(Nx * xspacing - spacing) / 2., -(Ny * yspacing - spacing * np.tan(np.deg2rad(60))) / 2.))
        shapearray2 = CellArray(shape, Nx, Ny / 2, (xspacing, yspacing * 2.), origin=(
            xspacing / 2. - (Nx * xspacing - spacing) / 2.,
            yspacing - (Ny * yspacing - spacing * np.tan(np.deg2rad(60))) / 2.))

        allshapes = Cell('All Shapes')
        allshapes.add(shapearray)
        allshapes.add(shapearray2)
        #            allshapes.add(shape)
        self.add(allshapes)

    def makeXShape(self, length, width, rotAngle, spacing, Nx, Ny, layers):
        if not (type(layers) == list): layers = [layers]
        slit = Cell("Slit")
        for l in layers:
            membrane = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=l)
            membrane_cell = Cell('Membrane_w{:.0f}'.format(width * 1000))
            membrane_cell.add(membrane)
            slit.add(membrane_cell)
        shape = Cell('Shapes')
        shape.add(slit, rotation=60)
        shape.add(slit, rotation=120)

        xspacing = (length + spacing) * np.cos(np.deg2rad(60))
        yspacing = (length + spacing) * np.sin(np.deg2rad(60))
        shapearray = CellArray(shape, Nx, Ny, (xspacing, yspacing),
                               origin=(-(Nx * xspacing - spacing) / 2., -(Ny * yspacing - spacing) / 2.))
        #            shapearray2 = CellArray(shape, Nx, Ny/2,(xspacing,yspacing*2.),origin=(xspacing/2.-(Nx*xspacing-spacing)/2.,yspacing-(Ny*yspacing-spacing*np.tan(np.deg2rad(60)))/2.))
        #            shapearray = CellArray(shape, Nx, Ny,(xspacing,yspacing))
        #            shapearray.rotate(rotAngle)
        #            shapearray.translate((-shapearray.bounding_box.mean(0)[0]/2.,-shapearray.bounding_box.mean(0)[1]/2.))

        allshapes = Cell('All Shapes')
        allshapes.add(shapearray)
        #            allshapes.add(shapearray2)
        #            allshapes.add(shape)
        self.add(allshapes)

    def makeArrowShape(self, length, width, rotAngle, spacing, Nx, Ny, layers):
        if not (type(layers) == list): layers = [layers]
        pt1 = np.array((-width * 0.3, -width / 2.))
        pt2 = np.array((length, width / 2.))
        slit = Cell("Slit")
        for l in layers:
            rect = Rectangle(pt1, pt2, layer=l)
            slit.add(rect)
            shape = Cell('Shapes')
            shape.add(slit, rotation=-120)
            shape.add(slit, rotation=120)

            xspacing = (width + spacing) / np.cos(np.deg2rad(30))
            yspacing = (length + spacing / 2.) * np.sin(np.deg2rad(60))
            shapearray = CellArray(shape, Nx, Ny, (xspacing, yspacing * 2.),
                                   origin=(-(Nx * xspacing - spacing) / 2., -(Ny * yspacing - spacing) / 2.))

            allshapes = Cell('All Shapes')
            allshapes.add(shapearray)
            #            allshapes.add(shapearray2)
            #            allshapes.add(shape)
            self.add(allshapes)


def generate_lattice(image_shape, lattice_vectors):
    """
    Generates points on a lattice defined by two lattice vectors that fall within rectangular region
    :param image_shape: list with two entries: length and width of desired area
    :param lattice_vectors: list of two vectors of length 2 which define the lattice
    :return: list of points
    """
    # Taken from https://stackoverflow.com/questions/6141955/efficiently-generate-a-lattice-of-points-in-python
    center_pix = np.array(image_shape) // 2
    # Get the lower limit on the cell size.
    dx_cell = max(abs(lattice_vectors[0][0]), abs(lattice_vectors[1][0]))
    dy_cell = max(abs(lattice_vectors[0][1]), abs(lattice_vectors[1][1]))
    # Get an over estimate of how many cells across and up.
    nx = image_shape[0] // dx_cell
    ny = image_shape[1] // dy_cell
    # Generate a square lattice, with too many points.
    # Here I generate a factor of 4 more points than I need, which ensures
    # coverage for highly sheared lattices.  If your lattice is not highly
    # sheared, than you can generate fewer points.
    x_sq = np.arange(-nx, nx, dtype=float)
    y_sq = np.arange(-ny, nx, dtype=float)
    x_sq.shape = x_sq.shape + (1,)
    y_sq.shape = (1,) + y_sq.shape
    # Now shear the whole thing using the lattice vectors
    x_lattice = lattice_vectors[0][0] * x_sq + lattice_vectors[1][0] * y_sq
    y_lattice = lattice_vectors[0][1] * x_sq + lattice_vectors[1][1] * y_sq
    # Trim to fit in box.
    mask = ((x_lattice < image_shape[0] / 2.0) & (x_lattice > -image_shape[0] / 2.0))
    mask = mask & ((y_lattice < image_shape[1] / 2.0) & (y_lattice > -image_shape[1] / 2.0))
    x_lattice = x_lattice[mask]
    y_lattice = y_lattice[mask]
    # Translate to the centre pix.
    x_lattice += center_pix[0]
    y_lattice += center_pix[1]
    # Make output compatible with original version.
    out = np.empty((len(x_lattice), 2), dtype=float)
    out[:, 0] = y_lattice
    out[:, 1] = x_lattice
    return out


def slit_elongation_array(pitches, spacing, widths, lengths, rot_angle, array_height, array_spacing, layers):
    if not (type(layers) == list):
        layers = [layers]
    if not (type(pitches) == list):
        pitches = [pitches]
    if not (type(lengths) == list):
        lengths = [lengths]
    if not (type(widths) == list):
        widths = [widths]
    for l in layers:
        j = -1
        manyslits = Cell("SlitArray")
        slitarray = Cell("SlitArray")
        pitch = pitches[0]
        width = widths[0]
        j += 1
        i = -1
        x_length = 0
        slit = Cell("Slits")
        for length in lengths:
            spacing = length / 5. + 0.1
            i += 1
            pitch_v = pitch / np.cos(np.deg2rad(rot_angle))
            n_y = int(array_height / pitch_v)
            # Define the slits
            if x_length == 0:
                translation = (length / 2., 0)
                x_length += length
            else:
                translation = (x_length + spacing + length / 2., 0)
                x_length += length + spacing
            pt1 = np.array((-length / 2., -width / 2.)) + translation
            pt2 = np.array((length / 2., width / 2.)) + translation
            rect = Rectangle(pt1, pt2, layer=l)
            rect = rect.copy().rotate(rot_angle)
            slit.add(rect)
        slits = CellArray(slit, 1, n_y, (0, pitch_v))
        slits.translate((-slits.bounding_box[1, 0] / 2., -slits.bounding_box[1, 1] / 2.))

        slitarray.add(slits)
        text = Label('w/p\n%i/%i' % (width * 1000, pitch * 1000), 2, layer=l_smBeam)
        lbl_vert_offset = 1.4
        text.translate(tuple(
            np.array(-text.bounding_box.mean(0)) + np.array(
                (0, array_height / lbl_vert_offset))))  # Center justify label
        slitarray.add(text)
        manyslits.add(slitarray)
    return manyslits


def make_slit_array(x_vars, y_vars, stat_vars, var_names, spacing, rot_angle,
                    array_height, array_width, array_spacing, layers):
    if len(var_names) != 3:
        raise Exception('Error! Need to have three variable names.')
    if not (type(layers) == list):
        layers = [layers]
    if not (type(x_vars) == list):
        x_vars = [x_vars]
    if not (type(y_vars) == list):
        y_vars = [y_vars]
    if not (type(stat_vars) == list):
        stat_vars = [stat_vars]

    x_var_name = var_names[0]
    y_var_name = var_names[1]
    stat_var_name = var_names[2]

    for l in layers:
        j = -1
        manyslits = Cell("SlitArray")
        for x_var in x_vars:
            j += 1
            i = -1
            for y_var in y_vars:
                i += 1
                if i % 3 == 0:
                    j += 1  # Move to array to next line
                    i = 0  # Restart at left

                var_dict = {x_var_name: x_var, y_var_name: y_var, stat_var_name: stat_vars[0]}
                pitch = var_dict['pitch']
                width = var_dict['width']
                length = var_dict['length']

                pitch_v = pitch / np.cos(np.deg2rad(rot_angle))
                #                    widthV = width / np.cos(np.deg2rad(rotAngle))
                n_x = int(array_width / (length + spacing))
                n_y = int(array_height / pitch_v)
                # Define the slits
                slit = Cell("Slits")
                rect = Rectangle(
                    (-length / 2., -width / 2.),
                    (length / 2., width / 2.),
                    layer=l)
                rect = rect.copy().rotate(rot_angle)
                slit.add(rect)
                slits = CellArray(slit, n_x, n_y,
                                  (length + spacing, pitch_v))
                slits.translate((-(n_x - 1) * (length + spacing) / 2., -(n_y - 1) * pitch_v / 2.))
                slit_array = Cell("SlitArray")
                slit_array.add(slits)
                text = Label('w/p/l\n%i/%i/%i' % (width * 1000, pitch * 1000, length * 1000), 2, layer=l_smBeam)
                lbl_vert_offset = 1.35
                if j % 2 == 0:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            0, -array_height / lbl_vert_offset))))  # Center justify label
                else:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            0, array_height / lbl_vert_offset))))  # Center justify label
                slit_array.add(text)
                manyslits.add(slit_array,
                              origin=((array_width + array_spacing) * i, (
                                      array_height + 2. * array_spacing) * j - array_spacing / 2.))
    return manyslits


def make_branch_array(x_vars, y_vars, stat_vars, var_names, spacing, rot_angle,
                      array_height, array_width, array_spacing, layers):
    array_angle = 25  # 60 for 111
    if len(var_names) != 3:
        raise Exception('Error! Need to have three variable names.')
    if not (type(layers) == list):
        layers = [layers]
    if not (type(x_vars) == list):
        x_vars = [x_vars]
    if not (type(y_vars) == list):
        y_vars = [y_vars]
    if not (type(stat_vars) == list):
        stat_vars = [stat_vars]

    x_var_name = var_names[0]
    y_var_name = var_names[1]
    stat_var_name = var_names[2]

    for l in layers:
        j = -1
        manybranches = Cell("ManyBranches")
        for x_var in x_vars:
            j += 1
            i = -1
            for y_var in y_vars:
                i += 1
                if i % 3 == 0:
                    j += 1  # Move to array to next line
                    i = 0  # Restart at left

                var_dict = {x_var_name: x_var, y_var_name: y_var, stat_var_name: stat_vars[0]}
                pitch = var_dict['pitch']
                width = var_dict['width']
                length = var_dict['length']
                branch = make_branch(length, width, layers, rot_angle=rot_angle)

                # x_spacing = (length + pitch) * np.cos(np.deg2rad(array_angle))
                # y_spacing = (length + pitch) * np.sin(np.deg2rad(array_angle))
                # n_x = int(array_width / x_spacing)
                # n_y = int(array_width / y_spacing)
                # shape_array = CellArray(branch, n_x, n_y, (x_spacing, y_spacing), origin=(
                #     -(n_x * x_spacing - pitch * np.cos(np.deg2rad(array_angle))) / 2.,
                #     -(n_y * y_spacing - pitch * np.sin(np.deg2rad(array_angle))) / 2.))

                branch_array = Cell('BranchArray-{:.0f}/{:.0f}/{:.2f}-wpl'.format(width, spacing, length))
                # branch_array.add(shape_array)

                dir1 = pitch
                dir2 = length + 2 * pitch
                pts = generate_lattice([array_height, array_width], [[dir1, dir2], [dir2, dir1]])
                for pt in pts:
                    pt += np.array([-array_width / 2., -array_height / 2.])
                    branch_array.add(branch, origin=pt)

                if length * 1000 % 1000 == 0:
                    text = Label('w/p/l\n{:.0f}/{:.0f}/{:.0f}'.format(width * 1000, pitch * 1000, length), 2,
                                 layer=l_smBeam)
                elif length * 1000 % 100 == 0:
                    text = Label('w/p/l\n{:.0f}/{:.0f}/{:.1f}'.format(width * 1000, pitch * 1000, length), 2,
                                 layer=l_smBeam)
                else:
                    text = Label('w/p/l\n{:.0f}/{:.0f}/{:.2f}'.format(width * 1000, pitch * 1000, length), 2,
                                 layer=l_smBeam)

                lbl_vert_offset = 1.35
                if j % 2 == 0:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            0, -array_height / lbl_vert_offset))))  # Center justify label
                else:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            0, array_height / lbl_vert_offset))))  # Center justify label
                branch_array.add(text)
                manybranches.add(branch_array,
                                 origin=((array_width + array_spacing) * i, (
                                         array_height + 2. * array_spacing) * j - array_spacing / 2.))
    return manybranches


def make_rotating_slits(length, width, N, radius, layers, angle_ref=None, angle_sweep=360):
    """

    :param length: Length of the slits in the circle
    :param width: Width of the slits in the circle
    :param N: Number of slits going around the circle
    :param radius: Radius of the circle
    :param layers: Layers to write the slits in
    :param angle_ref: if None, no angle reference lines are added. If '111' then add reference lines at 30/60 degrees. If '100' then add reference lines at 45/90 degrees.
    :return:
    """
    cell = Cell('RotatingSlits')
    if not (type(layers) == list): layers = [layers]
    allslits = Cell('All Slits')
    angles = np.linspace(0, angle_sweep, N)
    #        radius = length*12.
    translation = (radius, 0)
    for l in layers:
        membrane = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=l)
        membrane_cell = Cell('Membrane_w{:.0f}'.format(width * 1000))
        membrane_cell.add(membrane)
        slit = Cell("Slit")
        slit.add(membrane_cell, origin=translation)
        for angle in angles:
            allslits.add(slit, rotation=angle)
        cell.add(allslits)

        if angle_ref:
            labelCell = Cell('AngleLabels')
            lineCell = Cell('Line')
            pt1 = (-radius * 0.9, 0)
            pt2 = (radius * 0.9, 0)
            line = Path([pt1, pt2], width=width, layer=l)
            dLine = dashed_line(pt1, pt2, 2, width, l)
            lineCell.add(line)
            labelCell.add(lineCell, rotation=0)
            if angle_ref == '111':
                labelCell.add(lineCell, rotation=60)
                labelCell.add(lineCell, rotation=-60)
                labelCell.add(dLine, rotation=30)
                labelCell.add(dLine, rotation=90)
                labelCell.add(dLine, rotation=-30)
            elif angle_ref == '100':
                labelCell.add(lineCell, rotation=0)
                labelCell.add(lineCell, rotation=90)
                labelCell.add(dLine, rotation=45)
                labelCell.add(dLine, rotation=135)
            cell.add(labelCell)

        return cell


def make_arm(width, length, layer, cell_name='branch'):
    membrane = Path([(0, 0), (length, 0)], width=width, layer=layer)
    membrane_cell = Cell('Membrane_w{:.0f}'.format(width * 1000))
    membrane_cell.add(membrane)
    cell = Cell(cell_name)
    cell.add(membrane_cell)
    return cell


def make_branch(length, width, layers, rot_angle=0):
    slit = Cell("Slit")
    for l in layers:
        membrane = Path([(0, 0), (length, 0)], width=width, layer=l)
        membrane_cell = Cell('Membrane_w{:.0f}'.format(width * 1000))
        membrane_cell.add(membrane)
        slit.add(membrane_cell)

    branch = Cell('Branch-{:.0f}/{:.2f}-wl'.format(width, length))
    branch.add(slit, rotation=0 + rot_angle)
    # branch.add(slit, rotation=90 + rot_angle)
    branch.add(slit, rotation=180 + rot_angle)
    branch.add(slit, rotation=270 + rot_angle)
    return branch


def make_branch_device(width, pitch, len_inner, len_outer, n_membranes, layer):
    branch_device = Cell('branch_device')
    inner_arm = make_arm(width, len_inner, layer, cell_name='inner_arm')
    outer_arm = make_arm(width, len_outer, layer, cell_name='outer_arm')
    outer_branch = Cell('outer_branch')
    outer_branch.add(outer_arm)
    outer_branch.add(outer_arm, rotation=90)

    branch_quarter = Cell('branch_third')
    branch_quarter.add(inner_arm)
    for i in range(1, int(n_membranes) + 1):
        branch_quarter.add(outer_branch, origin=((width + pitch) * i, (width + pitch) * i))

    # branch_device.add(branch_quarter, rotation=0)
    # branch_device.add(branch_quarter, rotation=90)
    branch_device.add(outer_arm, rotation=0)
    branch_device.add(branch_quarter, rotation=180)
    branch_device.add(branch_quarter, rotation=270)

    for i in range(1, int(n_membranes) + 1):
        length = (len_inner + i * pitch) * 2
        membrane = make_arm(width, length, layer, cell_name='outer_arms_top' + str(i))
        branches_top = Cell('branch_third')
        branches_top.add(membrane, origin=(-length / 2., (width + pitch) * i))
        branch_device.add(branches_top)

    return branch_device


def make_rotating_branch_devices(length, width, N, radius, layers, angle_sweep=360, angle_ref=False, angle_offset=0):
    cell = Cell('RotatingBranches')
    if not (type(layers) == list):
        layers = [layers]
    allslits = Cell('All Slits')
    angles = np.linspace(0, angle_sweep, N)
    translation = (radius, 0)

    branch_dev = make_branch_device(width, 0.750, length, length, 3, layers[0])
    for element in branch_dev.elements:
        element.origin = [radius, 0]
    for angle in angles:
        allslits.add(branch_dev.copy(), rotation=angle_offset + angle)
    cell.add(allslits)

    for l in layers:
        if angle_ref:
            labelCell = Cell('AngleLabels')
            lineCell = Cell('Line')
            pt1 = (0, 0)
            pt2 = (radius * 0.9, 0)
            line = Path([pt1, pt2], width=width, layer=l)
            dLine = dashed_line(pt1, pt2, 2, width, l)
            lineCell.add(line)

            rot_angle = 0
            while True:
                if abs(rot_angle) > abs(angle_sweep):
                    break
                if abs(rot_angle) % 60 == 0:
                    labelCell.add(lineCell, rotation=rot_angle)
                if (abs(rot_angle) - 30) % 60 == 0:
                    labelCell.add(dLine, rotation=rot_angle)
                rot_angle += np.sign(angle_sweep) * 15
            cell.add(labelCell)
    return cell


def make_rotating_branches(length, width, N, radius, layers, angle_sweep=360, angle_ref=False, angle_offset=0):
    cell = Cell('RotatingBranches')
    if not (type(layers) == list):
        layers = [layers]
    allslits = Cell('All Slits')
    angles = np.linspace(0, angle_sweep, N)
    translation = (radius, 0)
    pt1 = np.array((-length / 2., -width / 2.)) + translation
    pt2 = np.array((length / 2., width / 2.)) + translation

    branch = make_branch(length, width, layers)
    for element in branch.elements:
        element.origin = [radius, 0]
    for angle in angles:
        allslits.add(branch.copy(), rotation=angle_offset + angle)
    cell.add(allslits)

    for l in layers:
        if angle_ref:
            labelCell = Cell('AngleLabels')
            lineCell = Cell('Line')
            pt1 = (0, 0)
            pt2 = (radius * 0.9, 0)
            line = Path([pt1, pt2], width=width, layer=l)
            dLine = dashed_line(pt1, pt2, 2, width, l)
            lineCell.add(line)

            rot_angle = 0
            while True:
                if abs(rot_angle) > abs(angle_sweep):
                    break
                if abs(rot_angle) % 60 == 0:
                    labelCell.add(lineCell, rotation=rot_angle)
                if (abs(rot_angle) - 30) % 60 == 0:
                    labelCell.add(dLine, rotation=rot_angle)
                rot_angle += np.sign(angle_sweep) * 15
            cell.add(labelCell)
    return cell


def make_shape_array(array_size, shape_area, shape_pitch, type, layer):
    num_of_shapes = int(np.ceil(array_size / shape_pitch))
    base_cell = Cell('Base')
    if type.lower() == "circles":
        circ_radius = np.sqrt(shape_area / np.pi)
        circ = Disk([0, 0], circ_radius, layer=layer)
        base_cell.add(circ)
    elif type.lower() == 'tris_down':
        triangle_side = np.sqrt(shape_area / np.sqrt(3) * 4)
        tri_shape = RegPolygon([0, 0], triangle_side, 3, layer=layer)
        tri_cell = Cell('Tri')
        tri_cell.add(tri_shape)
        base_cell.add(tri_cell, rotation=30)
    elif type.lower() == 'tris_up':
        triangle_side = np.sqrt(shape_area / np.sqrt(3) * 4)
        tri_shape = RegPolygon([0, 0], triangle_side, 3, layer=layer)
        tri_cell = Cell('Tri')
        tri_cell.add(tri_shape)
        base_cell.add(tri_cell, rotation=-30)
    elif type.lower() == 'hexagons':
        hex_side = np.sqrt(shape_area / 6. / np.sqrt(3) * 4)
        hex_shape = RegPolygon([0, 0], hex_side, 6, layer=layer)
        hex_cell = Cell('Hex')
        hex_cell.add(hex_shape)
        base_cell.add(hex_cell, rotation=0)

    shape_array = CellArray(base_cell, num_of_shapes, num_of_shapes, [shape_pitch, shape_pitch])
    shape_array_cell = Cell('Shape Array')
    shape_array_cell.add(shape_array)

    text = Label('{}'.format(type), 2, layer=l_smBeam)
    lblVertOffset = 0.8
    text.translate(
        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
            array_size / 2., array_size / lblVertOffset))))  # Center justify label

    shape_array_cell.add(text)

    return shape_array_cell


def make_theory_cell_3br():
    ''' Makes the theory cell and returns it as a cell'''
    # Growth Theory Slit Elongation
    pitch = [0.500]
    lengths = list(np.logspace(-3, 0, 20) * 8.0)  # Logarithmic
    widths = [0.044, 0.028, 0.016, 0.012, 0.008]
    # TheorySlitElong = Cell('LenWidthDependence')
    # arrayHeight = 20.
    # arraySpacing = 30.
    # spacing = 10.
    #
    # TheorySlitElong.add(
    #     slit_elongation_array(pitch, spacing, widths[0], lengths, 0., arrayHeight, arraySpacing, l_smBeam))
    # TheorySlitElong.add(
    #     slit_elongation_array(pitch, spacing, widths[1], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
    #     origin=(0, -30))
    # TheorySlitElong.add(
    #     slit_elongation_array(pitch, spacing, widths[2], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
    #     origin=(0, -60))
    # TheorySlitElong.add(
    #     slit_elongation_array(pitch, spacing, widths[3], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
    #     origin=(0, -90))
    # TheorySlitElong.add(
    #     slit_elongation_array(pitch, spacing, widths[4], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
    #     origin=(0, -120))

    # Length Dependence
    LenWidDep = Cell('LenWidDependence')
    pitch = [0.5]
    lengths = list(np.logspace(-2, 0, 9) * 4.0)  # Logarithmic
    widths = [0.060, 0.040, 0.020]
    arrayHeight = 20.
    arrayWidth = arrayHeight
    arraySpacing = 30.
    spacing = 0.5

    for i, length in enumerate(lengths):
        for j, width in enumerate(widths):
            LenWidDep.add(
                make_branch_array(pitch, width, length, ['pitch', 'width', 'length'], spacing, 0, arrayHeight,
                                  arrayWidth,
                                  arraySpacing, l_smBeam),
                origin=(i * 30, j * 30))

    # Make rotating slits
    shape_len = 3.
    shape_wid = 0.040
    N = 13  # number of shapes in the 120 degree arc
    N_rows = 2
    angle_sweep = 180
    wheel_rad = 45.
    wheel1 = Cell('RotDependence_LongSlits')
    for i in range(N_rows):
        angle_offset = (i % 2) * 7.5
        _angle_sweep = 180. - (i % 2) * 15.
        _N = int(N * _angle_sweep / angle_sweep) + (i % 2)
        angle_ref = False
        if i == 0:
            angle_ref = True
        wheel1.add(make_rotating_branch_devices(shape_len, shape_wid, _N, wheel_rad + i * shape_len * 3., l_smBeam,
                                                angle_sweep=_angle_sweep,
                                                angle_ref=angle_ref,
                                                angle_offset=angle_offset))

    wheel2 = Cell('RotDependence_ShortSlits')
    angle_sweep = -180
    wheel2.add(make_rotating_slits(2, 0.040, 91, 6. * 2, l_smBeam, angle_ref='100', angle_sweep=angle_sweep))
    for i in range(10):  # number of concentric rings to make
        wheel2.add(make_rotating_slits(2, 0.040, 91, (7.2 + i * 1.2) * 2, l_smBeam, angle_sweep=angle_sweep))

    # Pitch Dependence
    PitchDep = Cell('PitchDependence')
    # pitches = list(np.round(np.logspace(-1, 1, 10), 1))  # Logarithmic
    pitches = [0.500, 1.000, 2.000, 4.000]
    widths = [0.020, 0.040, 0.080, 0.140, 0.220, 0.320]
    length = [2.]
    arrayHeight = 20.
    arrayWidth = arrayHeight
    arraySpacing = 30.
    spacing = 0.5

    for j, width in enumerate(widths):
        for i, pitch in enumerate(reversed(pitches)):
            PitchDep.add(
                make_branch_array(pitch, width, length, ['pitch', 'width', 'length'], spacing, 0, arrayHeight,
                                  arrayWidth,
                                  arraySpacing, l_smBeam),
                origin=(j * 30, i * 30))
    # # Make arrays of various shapes
    # hexagon_array = make_shape_array(20, 0.02, 0.75, 'Hexagons', l_smBeam)
    # circles_array = make_shape_array(20, 0.02, 0.75, 'Circles', l_smBeam)
    # triangle_down_array = make_shape_array(20, 0.02, 0.75, 'Tris_down', l_smBeam)
    # triangle_up_array = make_shape_array(20, 0.02, 0.75, 'Tris_up', l_smBeam)

    # Merry Christmas!
    xmas_texts = [LineLabel('Merry Christmas!', 2, style='gothgbt', line_width=0.044, layer=l_smBeam),
                  LineLabel('Merry Christmas!', 2, style='italict', line_width=0.044, layer=l_smBeam),
                  LineLabel('Merry Christmas!', 2, style='scriptc', line_width=0.044, layer=l_smBeam),
                  LineLabel('Merry Christmas!', 2, style='scripts', line_width=0.044, layer=l_smBeam),
                  LineLabel('Merry Christmas!', 2, style='romanc', line_width=0.044, layer=l_smBeam),
                  LineLabel('Merry Christmas!', 2, style='romand', line_width=0.044, layer=l_smBeam)]

    xmax_cell = Cell('MerryChristmas')
    for i, xmas_text in enumerate(xmas_texts):
        xmas_text.translate(
            tuple(np.array(-xmas_text.bounding_box.mean(0)) + np.array([20, -80 - i * 10.])))  # Center justify label
        xmax_cell.add(xmas_text)

    TopCell = Cell('GrowthTheoryTopCell')
    TopCell.add(wheel1, origin=(-100., -60.))
    TopCell.add(wheel2, origin=(-100., -65.))
    TopCell.add(PitchDep, origin=(-200., -280.))
    # TopCell.add3(TheorySlitElong, origin=(-250., -50))
    TopCell.add(LenWidDep, origin=(-200., -50.))
    # TopCell.add(hexagon_array, origin=(-100., -50))
    # TopCell.add(circles_array, origin=(-75., -50.))
    # TopCell.add(triangle_down_array, origin=(-50., -50))
    # TopCell.add(triangle_up_array, origin=(-25., -50))
    TopCell.add(xmax_cell, origin=(0, 50))
    return TopCell


if __name__ == "__main__":
    TopCell = make_theory_cell_3br()
    # Add the copied cell to a Layout and save
    layout = Layout('LIBRARY')
    layout.add(TopCell)
    layout.save('GrowthTheoryCell_3BD.gds')
