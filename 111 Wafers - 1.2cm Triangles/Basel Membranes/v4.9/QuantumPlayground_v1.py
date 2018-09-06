# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:04:48 2016

@author: Martin Friedl
"""

# TODO: Add arrays of intersecting growth shapes

import numpy as np

from gdsCAD_v045.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_v045.shapes import Box, Rectangle, Label, LineLabel, Disk, RegPolygon
from gdsCAD_v045.templates111_branches import dashed_line

# Layers
l_smBeam = 0
l_lgBeam = 1

smFrameSize = 400.


# TODO: move labels of all fields to the top of the fields
# TODO: fix the right circle so the slits line up with the rotation marker lines

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


def makeYShapes(length, width, rotAngle, spacing, Nx, Ny, layers):
    if not (type(layers) == list): layers = [layers]
    pt1 = np.array((0, -width / 2.))
    pt2 = np.array((length, width / 2.))
    slit = Cell("Slit")
    for l in layers:
        rect = Rectangle(pt1, pt2, layer=l)
        slit.add(rect)
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
        return allshapes


def makeTriShapes(length, width, rotAngle, spacing, Nx, Ny, layers):
    if not (type(layers) == list): layers = [layers]
    pt1 = np.array((-length / 2. - width / 4., -width / 2.)) + np.array([0, np.tan(np.deg2rad(30)) * length / 2.])
    pt2 = np.array((length / 2. + width / 4., width / 2.)) + np.array([0, np.tan(np.deg2rad(30)) * length / 2.])
    slit = Cell("Slit")
    for l in layers:
        rect = Rectangle(pt1, pt2, layer=l)
        slit.add(rect)
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
        return allshapes


def makeArrowShape(length, width, rotAngle, spacing, Nx, Ny, layers):
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
        return allshapes


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


def make_arm(width, length, layer, cell_name='branch'):
    cell = Cell(cell_name)
    rect = Rectangle((0, -width / 2.), (length, width / 2.), layer=layer)
    cell.add(rect)
    return cell


def make_branch(length, width, layers, rot_angle=0):
    pt1 = np.array((0, -width / 2.))
    pt2 = np.array((length, width / 2.))

    slit = Cell("Slit")
    for l in layers:
        rect = Rectangle(pt1, pt2, layer=l)
        slit.add(rect)

    branch = Cell('Branch-{}/{}-lw'.format(length, width))
    branch.add(slit, rotation=0 + rot_angle)
    branch.add(slit, rotation=120 + rot_angle)
    branch.add(slit, rotation=240 + rot_angle)
    return branch


def make_branch_device(width, pitch, len_inner, len_outer, n_membranes, layer):
    branch_device = Cell('branch_device')
    inner_arm = make_arm(width, len_inner, layer, cell_name='inner_arm')
    outer_arm = make_arm(width, len_outer, layer, cell_name='outer_arm')
    outer_branch = Cell('outer_branch')
    outer_branch.add(outer_arm)
    outer_branch.add(outer_arm, rotation=120)

    branch_third = Cell('branch_third')
    branch_third.add(inner_arm)
    for i in range(1, int(n_membranes) + 1):
        branch_third.add(outer_branch, origin=(np.cos(np.deg2rad(60)) * (width + pitch) * i, (width + pitch) * i))

    branch_device.add(branch_third, rotation=0)
    branch_device.add(branch_third, rotation=120)
    branch_device.add(branch_third, rotation=240)

    # self.add(branch_device)

    return branch_device


def makeShape_X(pitch, length, width, n_buffers, layer):
    """
    Makes X shaped branched structures for quantum transport experiments
    :param pitch: Center-to-center distance between slits
    :param length: Length of the crossing slits
    :param width: Width of the slits
    :param n_buffers: Number of slits to put around (buffer) the crossing slits
    :param layer: GDS layer to put the pattern on
    :return: Cell with the desired pattern on it
    """
    longline = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=layer)
    shortline = Path([(0, 0), (length / 2., 0)], width=width, layer=layer)
    long_slit = Cell('LongSlit')
    long_slit.add(longline)
    short_slit = Cell('ShortSlit')
    short_slit.add(shortline)

    xShape = Cell('XShape')
    xShape.add(long_slit, rotation=60)
    xShape.add(long_slit, rotation=120)

    if n_buffers <= 0:
        return xShape

    sideBufferNM = Cell('X_SideBufferNM')
    sideBufferNM.add(short_slit, rotation=60)
    sideBufferNM.add(short_slit, rotation=-60)

    topBufferNM = Cell('X_TopBufferNM')
    topBufferNM.add(short_slit, rotation=60)
    topBufferNM.add(short_slit, rotation=120)

    for i in range(n_buffers):
        xShape.add(sideBufferNM, origin=((i + 1) * pitch / np.sin(np.deg2rad(60)), 0))
        xShape.add(sideBufferNM, origin=(-(i + 1) * pitch / np.sin(np.deg2rad(60)), 0), rotation=180)
        xShape.add(topBufferNM, origin=(0, (i + 1) * pitch / np.cos(np.deg2rad(60))))
        xShape.add(topBufferNM, origin=(0, -(i + 1) * pitch / np.cos(np.deg2rad(60))), rotation=180)

    return xShape


def makeShape_Star(pitch, length, width, n_buffers, layer):
    """
    Makes star shaped branched structures for quantum transport experiments
    :param pitch: Center-to-center distance between slits
    :param length: Length of the crossing slits
    :param width: Width of the slits
    :param n_buffers: Number of slits to put around (buffer) the crossing slits
    :param layer: GDS layer to put the pattern on
    :return: Cell with the desired pattern on it
    """
    longline = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=layer)
    shortline = Path([(0, 0), (length / 2., 0)], width=width, layer=layer)
    long_slit = Cell('LongSlit')
    long_slit.add(longline)
    short_slit = Cell('ShortSlit')
    short_slit.add(shortline)

    starShape = Cell('StarShape')
    starShape.add(long_slit, rotation=0)
    starShape.add(long_slit, rotation=60)
    starShape.add(long_slit, rotation=120)

    if n_buffers <= 0:
        return starShape

    bufferNM = Cell('Star_BufferNM')

    for i in range(n_buffers):
        bufferNM.add(short_slit, origin=(0, (i + 1) * pitch / np.cos(np.deg2rad(60))), rotation=60)
        bufferNM.add(short_slit, origin=(0, (i + 1) * pitch / np.cos(np.deg2rad(60))), rotation=120)

    starShape.add(bufferNM, rotation=0)
    starShape.add(bufferNM, rotation=60)
    starShape.add(bufferNM, rotation=120)
    starShape.add(bufferNM, rotation=180)
    starShape.add(bufferNM, rotation=240)
    starShape.add(bufferNM, rotation=300)

    return starShape


def makeShape_HashTag(hashtag_pitch, buffer_pitch, length, width, layer):
    """
    Makes hashtag-shaped branched structures for quantum transport experiments
    :param hashtag_pitch: Center-to-center distance between slits in the hashtag
    :param buffer_pitch: Center-to-center distance between the buffer membranes
    :param length: Length of the crossing slits
    :param width: Width of the slits
    :param n_buffers: Number of slits to put around (buffer) the crossing slits
    :param layer: GDS layer to put the pattern on
    :return: Cell with the desired pattern on it
    """
    dx_big = hashtag_pitch / 2. / np.tan(np.deg2rad(60))
    dy_big = hashtag_pitch / 2.
    dx_small = buffer_pitch / np.tan(np.deg2rad(60))
    dy_small = buffer_pitch

    longline = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=layer)
    x_dist = hashtag_pitch / np.sin(np.deg2rad(60))

    long_slit = Cell('LongSlit')
    long_slit.add(longline)

    hashtagShape = Cell('HashtagShape')
    hashtagShape.add(long_slit, origin=(-dx_big, -dy_big))
    hashtagShape.add(long_slit, origin=(dx_big, dy_big))
    hashtagShape.add(long_slit, origin=(-x_dist / 2., 0), rotation=60)
    hashtagShape.add(long_slit, origin=(x_dist / 2., 0), rotation=60)

    n_buffers = int(hashtag_pitch / buffer_pitch) - 1

    if n_buffers <= 0:
        return hashtagShape

    shortline = Path([(-x_dist / 2. + buffer_pitch, 0), (x_dist / 2. - buffer_pitch, 0)], width=width, layer=layer)
    short_slit = Cell('ShortSlit')
    short_slit.add(shortline)

    buffers = Cell("HashtagBuffers")

    if n_buffers % 2 == 1:  # odd number of buffer slits
        buffers.add(shortline)
        n_buffers -= 1

    for i in range(n_buffers / 2):
        buffers.add(short_slit,
                    origin=(dx_big - (i + 1) * dx_small,
                            dy_big - (i + 1) * dy_small))
        buffers.add(short_slit,
                    origin=(-dx_big + (i + 1) * dx_small,
                            -dy_big + (i + 1) * dy_small))

    # There is definitely a sexier way of implementing the buffers on the outside of the hashtag...
    # This is a bit of a hacky quick fix, not so elegant as the buffers extend out too much usually...
    bufferRow = Cell("BufferRow")
    len_buffer = x_dist - 2 * buffer_pitch
    bufferRow.add(buffers)
    bufferRow.add(buffers, origin=(x_dist / 2. + buffer_pitch + len_buffer / 2., 0))
    bufferRow.add(buffers, origin=(-x_dist / 2. - buffer_pitch - len_buffer / 2., 0))

    hashtagShape.add(bufferRow)
    hashtagShape.add(bufferRow, origin=(2 * dx_big, 2 * dy_big))
    hashtagShape.add(bufferRow, origin=(-2 * dx_big, -2 * dy_big))

    return hashtagShape


def makeShape_Window(frame_pitch, buffer_pitch, length, width, layer):
    """
    Makes window-shaped branched structures for quantum transport experiments
    :param frame_pitch: Center-to-center distance between slits in the window
    :param buffer_pitch: Center-to-center distance between the buffer membranes
    :param length: Length of the crossing slits
    :param width: Width of the slits
    :param n_buffers: Number of slits to put around (buffer) the crossing slits
    :param layer: GDS layer to put the pattern on
    :return: Cell with the desired pattern on it
    """
    dx_big = frame_pitch / 2. / np.tan(np.deg2rad(60))
    dy_big = frame_pitch / 2.
    dx_small = buffer_pitch / np.tan(np.deg2rad(60))
    dy_small = buffer_pitch

    longline = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=layer)
    x_dist = frame_pitch / np.sin(np.deg2rad(60))

    long_slit = Cell('LongSlit')
    long_slit.add(longline)

    windowShape = Cell('WindowShape')
    windowShape.add(long_slit, origin=(-dx_big, -dy_big))
    windowShape.add(long_slit, origin=(dx_big, dy_big))
    windowShape.add(long_slit, origin=(-3 * dx_big, -3 * dy_big))
    windowShape.add(long_slit, origin=(3 * dx_big, 3 * dy_big))
    windowShape.add(long_slit, origin=(-x_dist / 2., 0), rotation=60)
    windowShape.add(long_slit, origin=(x_dist / 2., 0), rotation=60)
    windowShape.add(long_slit, origin=(-3 * x_dist / 2., 0), rotation=60)
    windowShape.add(long_slit, origin=(3 * x_dist / 2., 0), rotation=60)

    n_buffers = int(frame_pitch / buffer_pitch) - 1

    buffers = Cell("WindowBuffers")

    if n_buffers <= 0:
        return windowShape

    shortline = Path([(-x_dist / 2. + buffer_pitch, 0), (x_dist / 2. - buffer_pitch, 0)], width=width, layer=layer)
    short_slit = Cell('ShortSlit')
    short_slit.add(shortline)

    if n_buffers % 2 == 1:  # odd number of buffer slits
        buffers.add(shortline)
        n_buffers -= 1

    for i in range(n_buffers / 2):
        buffers.add(short_slit,
                    origin=(dx_big - (i + 1) * dx_small,
                            dy_big - (i + 1) * dy_small))
        buffers.add(short_slit,
                    origin=(-dx_big + (i + 1) * dx_small,
                            -dy_big + (i + 1) * dy_small))

    bufferRow = Cell("BufferRow")
    len_buffer = x_dist - 2 * buffer_pitch
    v1 = (x_dist / 2. + buffer_pitch + len_buffer / 2., 0)
    bufferRow.add(buffers)
    bufferRow.add(buffers, origin=(v1[0], v1[1]))
    bufferRow.add(buffers, origin=(-v1[0], -v1[1]))

    v2 = (2 * dx_big, 2 * dy_big)
    windowShape.add(bufferRow)
    windowShape.add(bufferRow, origin=(v2[0], v2[1]))
    windowShape.add(bufferRow, origin=(-v2[0], -v2[1]))
    windowShape.add(bufferRow, origin=(2 * v2[0], 2 * v2[1]))
    windowShape.add(bufferRow, origin=(-2 * v2[0], -2 * v2[1]))

    bufferCol = Cell("BufferCol")
    bufferCol.add(buffers)
    bufferCol.add(buffers, origin=(v2[0], v2[1]))
    bufferCol.add(buffers, origin=(-v2[0], -v2[1]))
    bufferCol.add(buffers, origin=(2 * v2[0], 2 * v2[1]))
    bufferCol.add(buffers, origin=(-2 * v2[0], -2 * v2[1]))
    windowShape.add(bufferCol, origin=(2 * v1[0], 2 * v1[1]))
    windowShape.add(bufferCol, origin=(-2 * v1[0], -2 * v1[1]))

    return windowShape


def makeShape_Triangle(pitch, length, width, contactlength, n_outer_buf, layer):
    """
    Makes triangle shape
    :param pitch: pitch of nanomembranes
    :param length: length of triangle side
    :param width: width of the membranes
    :param contactlength: length of membrane coming from triangle to contact
    :param n_outer_buf: number of outer buffers to add
    :param layer: layer to write the structures in
    :return: cell with triangle shape in it
    """
    triangleshape = Cell("TriangleShape")

    line = Cell('TriLine')

    height = length * np.sqrt(3.) / 2.
    pt1 = (-length / 2., 0)
    pt2 = (length / 2., 0)
    pt3 = (0, height)
    centroid = (np.mean([pt1[0], pt2[0], pt3[0]]), np.mean([pt1[1], pt2[1], pt3[1]]))

    tri_arm = Cell('TriangleArms')

    # Make the main triangle with contact area
    line_cell = Cell('LineCell')
    line = Path([(-length / 2. - contactlength, 0), (length / 2., 0)], width=width, layer=layer)
    line_cell.add(line)
    tri_arm.add(line_cell, origin=(0, 0))

    n_inner_buf = int(centroid[1] / pitch) + 1
    print n_inner_buf

    # Make the inner buffer triangles
    for i in range(1, n_inner_buf):
        line_cell = Cell('LineCell')
        slit_len = length - 2 * pitch / np.tan(np.deg2rad(30)) * i
        line = Path([(-slit_len / 2., 0), (slit_len / 2., 0)], width=width, layer=layer)
        line_cell.add(line)
        tri_arm.add(line_cell, origin=(0, -i * pitch))

    # Make the outer buffer membranes
    for i in range(1, n_outer_buf + 1):
        line_cell = Cell('LineCell')
        line = Path([(-length / 2. - contactlength - pitch / np.tan(np.deg2rad(30)) * i, 0),
                     (length / 2. + pitch / np.tan(np.deg2rad(30)) * i - pitch / np.cos(np.deg2rad(30)) * (i + 1), 0)],
                    width=width, layer=layer)
        line_cell.add(line)
        tri_arm.add(line_cell, origin=(0, i * pitch))

    tri_arm_offcenter = Cell('TriArmOffCenter')
    tri_arm_offcenter.add(tri_arm, origin=centroid)

    triangleshape.add(tri_arm_offcenter, rotation=0)
    triangleshape.add(tri_arm_offcenter, rotation=120)
    triangleshape.add(tri_arm_offcenter, rotation=240)

    return triangleshape


def makeShape_ABDiamond(pitch, length, width, contactlength, n_outer_buf, layer):
    """
    Makes Aharonov Bohm diamond shape
    :param pitch: pitch of nanomembranes
    :param length: quadrilateral side length
    :param width: width of the membranes
    :param contactlength: length of membrane coming from triangle to contact
    :param n_outer_buf: number of outer buffers to add
    :param layer: layer to write the structures in
    :return: cell with triangle shape in it
    """
    diamondshape = Cell("DiamondShape")

    line = Cell('DiamondLine')

    height = length * np.sqrt(3.) / 2.
    pt1 = (-length / 2., 0)
    pt2 = (length / 2., 0)

    shape_half = Cell('DiamondShapeHalf')

    # Make the main quadrilateral with contact area
    line_cell = Cell('LineCell')
    contactline_cell = Cell('ContactLineCell')
    line = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=layer)
    contactline = Path([(-length / 2., 0), (length / 2. + contactlength, 0)], width=width, layer=layer)
    line_cell.add(line)
    contactline_cell.add(contactline)
    shape_half.add(line_cell, origin=(-length / 2., 0), rotation=60)
    shape_half.add(contactline_cell, origin=(length / 4., height / 2.))

    n_inner_buf = int(height / 2. / pitch) + 1
    print 'n_buff'
    print n_inner_buf

    # Make the inner buffer triangles
    for i in range(1, n_inner_buf):
        line_cell = Cell('LineCell')
        slit_len = length - 2 * pitch * 2. / np.sqrt(3) * i
        line = Path([(-slit_len / 2., 0), (slit_len / 2., 0)], width=width, layer=layer)
        line_cell.add(line)
        shape_half.add(line_cell, origin=(length / 4. - pitch / np.sqrt(3) * i, height / 2 - i * pitch))
        shape_half.add(line_cell, origin=(-length / 2. + pitch * 2. / np.sqrt(3) * i, 0), rotation=60)

    # Make the outer buffer membranes
    for i in range(1, n_outer_buf + 1):
        line_cell = Cell('LineCell')
        contactline_cell = Cell('ContactBufferCell')
        slit_len = length + 2 * pitch * 2. / np.sqrt(3) * i
        line = Path([(-slit_len / 2. + 2 * pitch / np.sqrt(3) * (i + 1), 0), (slit_len / 2., 0)], width=width,
                    layer=layer)
        contactline = Path([(-slit_len / 2., 0), (slit_len / 2., 0)], width=width, layer=layer)
        line_cell.add(line)
        contactline_cell.add(contactline)
        shape_half.add(line_cell, origin=(-length / 2. - pitch * 2. / np.sqrt(3) * i, 0), rotation=60)
        shape_half.add(contactline_cell, origin=(length / 4. + pitch / np.sqrt(3) * i, height / 2 + i * pitch))

    diamondshape.add(shape_half, rotation=0)
    diamondshape.add(shape_half, rotation=180)

    return diamondshape


def makeShape_ABHexagon(pitch, length, width, contactlength, n_outer_buf, layer):
    """
    Makes Aharonov Bohm hexagon shape
    :param pitch: pitch of nanomembranes
    :param length: hexagon side length
    :param width: width of the membranes
    :param contactlength: length of membrane coming from triangle to contact
    :param n_outer_buf: number of outer buffers to add
    :param layer: layer to write the structures in
    :return: cell with triangle shape in it
    """
    hexagonshape = Cell("HexagonShape")

    line = Cell('HexagonLine')

    height = length * np.sqrt(3.) / 2.
    pt1 = (-length / 2., 0)
    pt2 = (length / 2., 0)

    shape_half = Cell('HexagonShapeHalf')

    # Make the main quadrilateral with contact area
    line_cell = Cell('LineCell')
    contactline_cell = Cell('ContactLineCell')
    line = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=layer)
    contactline = Path([(-length / 2., 0), (length / 2. + contactlength, 0)], width=width, layer=layer)
    line_cell.add(line)
    contactline_cell.add(contactline)
    shape_half.add(line_cell, origin=(-3 * length / 4., height / 2.), rotation=60)
    shape_half.add(line_cell, origin=(3 * length / 4., height / 2.), rotation=-60)
    shape_half.add(contactline_cell, origin=(0, height))

    n_inner_buf = int(height / pitch) + 1

    # Make the inner buffer hexagons
    for i in range(1, n_inner_buf):
        line_cell = Cell('LineCell')
        slit_len = length - 2 * pitch / np.sqrt(3) * i
        line = Path([(-slit_len / 2., 0), (slit_len / 2., 0)], width=width, layer=layer)
        line_cell.add(line)
        shape_half.add(line_cell, origin=(-3 * slit_len / 4., (height - (i * pitch)) / 2.), rotation=60)
        shape_half.add(line_cell, origin=(3 * slit_len / 4., (height - (i * pitch)) / 2.), rotation=-60)
        shape_half.add(line_cell, origin=(0, height - (i * pitch)))

    # Make the outer buffer membranes
    for i in range(1, n_outer_buf + 1):
        line_cell = Cell('LineCell')
        contactline_cell = Cell('ContactLineBuffer')
        shortline_cell = Cell('ShortLineBuffer')
        slit_len = length + 2 * pitch / np.sqrt(3) * i
        line = Path([(-slit_len / 2., 0), (slit_len / 2., 0)], width=width, layer=layer)
        shortline = Path([(2 * pitch / np.sqrt(3) * (i + 1) - slit_len / 2., 0), (slit_len / 2., 0)], width=width,
                         layer=layer)
        contactline = Path([(-slit_len / 2., 0), (slit_len / 2. + contactlength, 0)], width=width, layer=layer)
        line_cell.add(line)
        contactline_cell.add(contactline)
        shortline_cell.add(shortline)
        shape_half.add(line_cell, origin=(-3 * slit_len / 4., (height + (i * pitch)) / 2.), rotation=60)
        shape_half.add(shortline_cell, origin=(3 * slit_len / 4., (height + (i * pitch)) / 2.), rotation=-60)
        shape_half.add(contactline_cell, origin=(0, height + (i * pitch)))

    hexagonshape.add(shape_half, rotation=0)
    hexagonshape.add(shape_half, rotation=180)

    return hexagonshape


def make_qp():
    ''' Makes the theory cell and returns it as a cell'''
    widths = [0.028, 0.044]
    qp_spacing = 60

    TopCell = Cell('QuantumPlayground_TopCell')

    for i, nm_width in enumerate(widths):
        cell_XShape = makeShape_X(1.0, 10, nm_width, 3, l_smBeam)
        cell_XShapeNoBuffer = makeShape_X(1.0, 10, nm_width, 0, l_smBeam)
        cell_StarShape = makeShape_Star(1.0, 10, nm_width, 3, l_smBeam)
        cell_StarShapeNoBuffer = makeShape_Star(1.0, 10, nm_width, 0, l_smBeam)
        cell_HashTag = makeShape_HashTag(3.0, 1.0, 5, nm_width, l_smBeam)
        cell_HashTagNoBuffer = makeShape_HashTag(3.0, 10.0, 5, nm_width, l_smBeam)
        cell_Window = makeShape_Window(3.0, 1.0, 12, nm_width, l_smBeam)
        cell_WindowNoBuffer = makeShape_Window(3.0, 100.0, 12, nm_width, l_smBeam)
        cell_Triangle = makeShape_Triangle(1.0, 5, nm_width, 1.0, 3, l_smBeam)
        cell_TriangleNoBuffer = makeShape_Triangle(10.0, 5, nm_width, 1.0, 0, l_smBeam)
        cell_AB_Diamond = makeShape_ABDiamond(1.0, 3, nm_width, 1.0, 3, l_smBeam)
        cell_AB_DiamondNoBuffer = makeShape_ABDiamond(10.0, 3, nm_width, 1.0, 0, l_smBeam)
        cell_AB_Hexagon = makeShape_ABHexagon(1.0, 2, nm_width, 2.0, 3, l_smBeam)
        cell_AB_HexagonNoBuffer = makeShape_ABHexagon(10.0, 2, nm_width, 2.0, 0, l_smBeam)

        QP = Cell('QuantumPlayground_W{:.0f}'.format(nm_width * 1000))
        QP.add(cell_AB_Diamond, origin=(-30, 30))
        QP.add(cell_AB_DiamondNoBuffer, origin=(-10, 30))
        QP.add(cell_XShape, origin=(-30, 10))
        QP.add(cell_XShapeNoBuffer, origin=(-10, 10))
        QP.add(cell_HashTag, origin=(-30, -10))
        QP.add(cell_HashTagNoBuffer, origin=(-10, -10))
        QP.add(cell_Triangle, origin=(-30, -30))
        QP.add(cell_TriangleNoBuffer, origin=(-10, -30))
        QP.add(cell_AB_Hexagon, origin=(10, 30))
        QP.add(cell_AB_HexagonNoBuffer, origin=(30, 30))
        QP.add(cell_StarShape, origin=(10, 10))
        QP.add(cell_StarShapeNoBuffer, origin=(30, 10))
        QP.add(cell_Window, origin=(10, -10))
        QP.add(cell_WindowNoBuffer, origin=(30, -10))

        TopCell.add(QP, origin=(-qp_spacing + 2 * i * qp_spacing, qp_spacing))

        # SMALL Aharonov Bohm devices
        cell_Triangle_sm = makeShape_Triangle(1.0, 1, nm_width, 1.0, 5, l_smBeam)
        cell_TriangleNoBuffer_sm = makeShape_Triangle(10.0, 1, nm_width, 1.0, 0, l_smBeam)
        cell_AB_Diamond_sm = makeShape_ABDiamond(1.0, 1, nm_width, 1.0, 5, l_smBeam)
        cell_AB_DiamondNoBuffer_sm = makeShape_ABDiamond(10.0, 1, nm_width, 1.0, 0, l_smBeam)
        cell_AB_Hexagon_sm = makeShape_ABHexagon(1.0, 1, nm_width, 1.0, 5, l_smBeam)
        cell_AB_HexagonNoBuffer_sm = makeShape_ABHexagon(10.0, 1, nm_width, 1.0, 0, l_smBeam)

        QP_sm = Cell('QuantumPlayground_W{:.0f}'.format(nm_width * 1000))
        QP_sm.add(cell_AB_Diamond_sm, origin=(-30, 30))
        QP_sm.add(cell_AB_DiamondNoBuffer_sm, origin=(-10, 30))
        QP_sm.add(cell_XShape, origin=(-30, 10))
        QP_sm.add(cell_XShapeNoBuffer, origin=(-10, 10))
        QP_sm.add(cell_HashTag, origin=(-30, -10))
        QP_sm.add(cell_HashTagNoBuffer, origin=(-10, -10))
        QP_sm.add(cell_Triangle_sm, origin=(-30, -30))
        QP_sm.add(cell_TriangleNoBuffer_sm, origin=(-10, -30))
        QP_sm.add(cell_AB_Hexagon_sm, origin=(10, 30))
        QP_sm.add(cell_AB_HexagonNoBuffer_sm, origin=(30, 30))
        QP_sm.add(cell_StarShape, origin=(10, 10))
        QP_sm.add(cell_StarShapeNoBuffer, origin=(30, 10))
        QP_sm.add(cell_Window, origin=(10, -10))
        QP_sm.add(cell_WindowNoBuffer, origin=(30, -10))

        TopCell.add(QP_sm, origin=(-qp_spacing + 2 * i * qp_spacing, -qp_spacing))

    return TopCell


if __name__ == "__main__":
    TopCell = make_qp()
    # Add the copied cell to a Layout and save
    layout = Layout('LIBRARY')
    layout.add(TopCell)
    layout.save('QuantumPlayground_v1.0.gds')
