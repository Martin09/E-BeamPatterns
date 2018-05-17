# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2015

Pattern with individual membrane branches surrounded by scrificial membranes to make branched devices

@author: Martin Friedl
"""

# TODO: Finish cleaning up code warnings

# TODO: update the growth theory cell for the branching shapes (including rotating branches!)

from datetime import timedelta, date

import numpy as np
from gdsCAD.core import Cell, Boundary, CellArray, Layout
from gdsCAD.shapes import Box, Rectangle, Label
from shapely.affinity import rotate as rotateshape
from shapely.geometry import LineString

from GrowthTheoryCell_BranchDevices import make_theory_cell
from templates111_branches import Wafer_TriangStyle, dashed_line

putOnWafer = True  # Output full wafer or just a single pattern?
HighDensity = False  # High density of triangles?
glbAlignmentMarks = False
tDicingMarks = 10.  # Dicing mark line thickness (um)
rotAngle = 0.  # Rotation angle of the membranes
wafer_r = 20e3
waferVer = 'Membrane Branch Devices v1.0 r{:d}'.format(int(wafer_r / 1000))
mkWidthMinor = 3  # Width of minor (Basel) markers within each triangular chip

if HighDensity:
    CELL_GAP = 400
    density = 'HighDensity Basel Wafer - ' + date.today().strftime("%d%m%Y")
else:
    CELL_GAP = 3000
    density = 'LowDensity Basel Wafer - ' + date.today().strftime("%d%m%Y")
waferLabel = waferVer + '\n' + density
# Layers
l_smBeam = 0
l_lgBeam = 1


# %% Wafer template for MBE growth
class MBEWafer(Wafer_TriangStyle):
    """
    A 2" wafer divided into triangular cells
    """

    def __init__(self,
                 name,
                 cells=None,
                 wafer_r=20e3,
                 trisize=12e3,
                 cellsize=2e3,
                 block_gap=0.,  # 1200
                 cell_gap=800.,  # 800
                 doMCSearch=True,
                 MCIterations=100,
                 doMCBlockSearch=True,
                 MCBlockIterations=100,
                 mkWidth=10,
                 cellsAtEdges=False,
                 symmetric_chips=True):

        Wafer_TriangStyle.__init__(self,
                                   name,
                                   wafer_r=wafer_r,
                                   cells=cells,
                                   trisize=trisize,
                                   cellsize=cellsize,
                                   block_gap=block_gap,
                                   cell_gap=cell_gap,
                                   doMCSearch=doMCSearch,
                                   MCIterations=MCIterations,
                                   doMCBlockSearch=doMCBlockSearch,
                                   MCBlockIterations=MCBlockIterations,
                                   mkWidth=mkWidth,
                                   cellsAtEdges=cellsAtEdges,
                                   symmetric_chips=symmetric_chips)

        self.align_markers = None
        self.symmetric_chips = False
        # The placement of the wafer alignment markers
        am_x = 1.5e4
        am_y = 1.5e4
        self.align_pts = np.array([am_x, am_y])
        self.align_pts = np.vstack((self.align_pts, self.align_pts *
                                    (-1, 1)))  # Reflect about y-axis
        self.align_pts = np.vstack((self.align_pts, self.align_pts *
                                    (1, -1)))  # Reflect about x-axis

        self.o_text = {'UR': (am_x + 0.1e4, am_y + 0.1e4),
                       'UL': (-am_x - 0.1e4, am_y + 0.1e4),
                       'LL': (-am_x - 0.1e4, -am_y - 0.1e4),
                       'LR': (am_x + 0.1e4, -am_y - 0.1e4)}

        self._place_blocks()
        if glbAlignmentMarks:
            self.add_aligment_marks(l_lgBeam)
            self.add_orientation_text(l_lgBeam)
        points = self.add_dicing_marks(l_lgBeam, mkWidth=mkWidth)  # Width of dicing marks
        self.make_basel_align_marks(points, l_lgBeam, mk_width=mkWidthMinor)
        self.add_wafer_outline(100)
        self.build_and_add_blocks()
        self.add_blockLabels(l_lgBeam, center=True)
        self.add_cellLabels(l_lgBeam, center=True)  # Put cell labels ('A','B','C'...) in center of each cell
        self.add_theory_cell()
        bottom = np.array([0, -self.wafer_r * 0.85])
        # Write label on layer 100 to saving writing time
        self.add_waferLabel(waferLabel, 100, pos=bottom)

    def add_theory_cell(self):

        theory_cell = make_theory_cell()
        theory_cells = Cell('TheoryCells')
        for x, y in self.upCenters:
            theory_cells.add(theory_cell, origin=(x, y - 1500))
        for x, y in self.downCenters:
            if self.symmetric_chips:
                theory_cells.add(theory_cell, origin=(x, y + 1500), rotation=180)
            theory_cells.add(theory_cell, origin=(x, y + 1500))
        self.add(theory_cells)

    def make_basel_align_marks(self, points, layers, mk_width=5):
        if not (type(layers) == list):
            layers = [layers]
        wafer_rad = self.wafer_r
        tri_height = np.sqrt(3.) / 2. * self.trisize
        # Shift the points from the old dicing lines to make the dashed dicing lines
        points1 = np.array(points) + (self.trisize / 2., 0.)
        points2 = np.array(points) + (self.trisize / 4., tri_height / 2.)
        new_pts = np.vstack((points1, points2))
        # Create a lineshape of the boundary of the circle
        c = self.waferShape.boundary
        # Create a set (unordered with unique entries)
        dicing_lines = set()
        # For each point in the lattice, create three lines (one along each direction)
        for x, y in new_pts:
            l0 = LineString([(-4. * wafer_rad, y), (4. * wafer_rad, y)])
            l1 = rotateshape(l0, 60, origin=(x, y))
            l2 = rotateshape(l0, -60, origin=(x, y))
            # See where these lines intersect the wafer outline
            i0 = c.intersection(l0)
            i1 = c.intersection(l1)
            i2 = c.intersection(l2)
            if not i0.geoms == []:
                p0s = tuple(map(tuple, np.round((i0.geoms[0].coords[0], i0.geoms[
                    1].coords[0]))))
                dicing_lines.add(p0s)  # Add these points to a unique unordered set
            if not i1.geoms == []:
                p1s = tuple(map(tuple, np.round((i1.geoms[0].coords[0], i1.geoms[
                    1].coords[0]))))
                dicing_lines.add(p1s)
            if not i2.geoms == []:
                p2s = tuple(map(tuple, np.round((i2.geoms[0].coords[0], i2.geoms[
                    1].coords[0]))))
                dicing_lines.add(p2s)

        hpoints1 = np.array(points) + (self.trisize / 8., tri_height / 4. + 300.)
        hpoints2 = np.array(points) + (-self.trisize / 8., -tri_height / 4. - 450.)
        hpoints = np.vstack((hpoints1, hpoints2))
        # Make horizontal lines to cleave each mini-triangle chip and make it fit in the 6x6mm chip holder in Basel
        for x, y in hpoints:
            l0 = LineString([(-4. * wafer_rad, y), (4. * wafer_rad, y)])
            # See where this line intersects the wafer outline
            i0 = c.intersection(l0)
            if not i0.geoms == []:
                p0s = tuple(map(tuple, np.round((i0.geoms[0].coords[0], i0.geoms[
                    1].coords[0]))))
                dicing_lines.add(p0s)  # Add these points to a unique unordered set

        # At the end of the loop, the set will contain a list of point pairs which can be used to make the dicing marks
        dmarks = Cell('DIC_MRKS')
        for l in layers:
            for p1, p2 in dicing_lines:
                # dicing_line = Path([p1, p2], width=mkWidth,layer=l)
                dicing_line = dashed_line(p1, p2, 500, mk_width, l)
                dmarks.add(dicing_line)
            self.add(dmarks)


class Frame(Cell):
    """
    Make a frame for writing to with ebeam lithography
    Params:
    -name of the frame, just like when naming a cell
    -size: the size of the frame as an array [xsize,ysize]
    """

    def __init__(self, name, size, border_layers):
        if not (type(border_layers) == list):
            border_layers = [border_layers]
        Cell.__init__(self, name)
        self.size_x, self.size_y = size
        # Create the border of the cell
        for l in border_layers:
            self.border = Box(
                (-self.size_x / 2., -self.size_y / 2.),
                (self.size_x / 2., self.size_y / 2.),
                1,
                layer=l)
            self.add(self.border)  # Add border to the frame

        self.align_markers = None

    def make_align_markers(self, t, w, position, layers, cross=False, auto_align_marks=True):
        if not (type(layers) == list):
            layers = [layers]
        self.align_markers = Cell("AlignMarks_Four")
        align_mark = Cell("AlignMarks_Corner")
        for l in layers:
            if not cross:
                am0 = Rectangle((-w / 2., -w / 2.), (w / 2., w / 2.), layer=l)
            elif cross:
                crosspts = [(0, 0), (w / 2., 0), (w / 2., t), (t, t), (t, w / 2), (0, w / 2), (0, 0)]
                crosspts.extend(tuple(map(tuple, (-np.array(crosspts)).tolist())))
                am0 = Boundary(crosspts, layer=l)  # Create gdsCAD shape
                align_cross = Cell('AlignCross')
                align_cross.add(am0)
                align_mark.add(align_cross)

            if auto_align_marks:
                aa_mark = Rectangle((-10, -10), (10, 10), layer=l)
                auto_mark = Cell('AutoAlignMark')
                auto_mark.add(aa_mark)
                four_marks = Cell('FourAAMarks')
                pos = [100., 100.]
                four_marks.add(auto_mark, origin=tuple(np.array(pos) * [1, 1]))
                four_marks.add(auto_mark, origin=tuple(np.array(pos) * [-1, 1]))
                four_marks.add(auto_mark, origin=tuple(np.array(pos) * [1, -1]))
                four_marks.add(auto_mark, origin=tuple(np.array(pos) * [-1, -1]))
                align_mark.add(four_marks)

            self.align_markers.add(align_mark, origin=tuple(np.array(position) * [1, 1]))
            self.align_markers.add(align_mark, origin=tuple(np.array(position) * [-1, 1]))
            self.align_markers.add(align_mark, origin=tuple(np.array(position) * [1, -1]))
            self.align_markers.add(align_mark, origin=tuple(np.array(position) * [-1, -1]))
            self.add(self.align_markers)

    def make_arm(self, width, length, layer, cell_name='branch'):
        cell = Cell(cell_name)
        rect = Rectangle((0, -width / 2.), (length, width / 2.), layer=layer)
        cell.add(rect)
        return cell

    def make_branch_device(self, width, pitch, len_inner, len_outer, n_membranes, layer):

        branch_device = Cell('branch_device')
        inner_arm = self.make_arm(width, len_inner, layer, cell_name='inner_arm')
        outer_arm = self.make_arm(width, len_outer, layer, cell_name='outer_arm')
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

    def make_branch_device_array(self, spacing, _widths, array_height, array_width,
                                 array_spacing, len_inner, len_outer, n_membranes, layers):
        if not (type(layers) == list):
            layers = [layers]
        if not (type(_widths) == list):
            _widths = [_widths]
        for l in layers:
            i = -1
            j = 0
            manydevices = Cell("ManyDevices")
            for width in _widths:
                device = self.make_branch_device(width, spacing, len_inner, len_outer, n_membranes, l)
                [[x_min, y_min], [x_max, y_max]] = device.bounding_box
                x_size = abs(x_max - x_min)
                y_size = abs(y_max - y_min)

                i += 1
                if i % 3 == 0:
                    j += 1  # Move to array to next line
                    i = 0  # Restart at left

                nx = int(array_width / (x_size + spacing))
                ny = int(array_height / (y_size + spacing))

                devices = CellArray(device, nx, ny, (x_size + spacing, y_size + spacing))
                devices.translate((-(nx - 1) * (x_size + spacing) / 2., -(ny - 1) * (y_size + spacing) / 2.))
                device_array = Cell("DeviceArray")
                device_array.add(devices)
                # Make the labels for each array of devices
                text = Label('w/s/l\n%i/%.1f/%i' % (width * 1000, spacing, len_outer), 5)
                lbl_vertical_offset = 1.40
                if j % 2 == 0:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            0, -array_height / lbl_vertical_offset))))  # Center justify label
                else:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            0, array_height / lbl_vertical_offset))))  # Center justify label
                # TODO: Finish this below
                device_array.add(text)
                manydevices.add(device_array, origin=(
                    (array_width + array_spacing) * i, (array_height + 2. * array_spacing) * j - array_spacing / 2.))

            self.add(manydevices,
                     origin=(-i * (array_width + array_spacing) / 2, -(j + 1.5) * (array_height + array_spacing) / 2))

    def make_slit_array(self, nm_spacings, spacing, _widths, _lengths, rot_angle,
                        array_height, array_width, array_spacing, layers):
        if not (type(layers) == list):
            layers = [layers]
        if not (type(nm_spacings) == list):
            nm_spacings = [nm_spacings]
        if not (type(_lengths) == list):
            _lengths = [_lengths]
        if not (type(_widths) == list):
            _widths = [_widths]
        manyslits = i = j = None
        for l in layers:
            i = -1
            j = -1
            manyslits = Cell("SlitArray")
            nm_spacing = nm_spacings[0]
            for length in _lengths:
                j += 1
                i = -1

                for width in _widths:
                    # for nm_spacing in pitches:
                    i += 1
                    if i % 3 == 0:
                        j += 1  # Move to array to next line
                        i = 0  # Restart at left

                    pitch_v = nm_spacing / np.cos(np.deg2rad(rot_angle))
                    #                    widthV = width / np.cos(np.deg2rad(rotAngle))
                    ny = nx = int(array_width / (length + nm_spacing))
                    # ny = int(array_height / pitch_v)
                    # Define the slits
                    slit = Cell("Slits")
                    rect = Rectangle(
                        (-length / 2., -width / 2.),
                        (length / 2., width / 2.),
                        layer=l)
                    rect = rect.copy().rotate(rot_angle)
                    slit.add(rect)
                    slits = CellArray(slit, nx, ny,
                                      (length + spacing, pitch_v))
                    slits.translate((-(nx - 1) * (length + spacing) / 2., -(ny - 1) * pitch_v / 2.))
                    slit_array = Cell("SlitArray")
                    slit_array.add(slits)
                    text = Label('w/s/l\n%i/%i/%i' %
                                 (width * 1000, nm_spacing, length), 5)
                    lbl_vertical_offset = 1.35
                    if j % 2 == 0:
                        text.translate(
                            tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                                0, -array_height / lbl_vertical_offset))))  # Center justify label
                    else:
                        text.translate(
                            tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                                0, array_height / lbl_vertical_offset))))  # Center justify label
                    # TODO: Finish this below
                    slit_array = self.branch_shape_array(length, width, 0, nm_spacing, nx, ny, l)
                    slit_array.add(text)
                    manyslits.add(slit_array,
                                  origin=((array_width + array_spacing) * i, (
                                      array_height + 2. * array_spacing) * j - array_spacing / 2.))

        self.add(manyslits,
                 origin=(-i * (array_width + array_spacing) / 2, -(j + 1.5) * (
                     array_height + array_spacing) / 2))

    def branch_shape_array(self, length, width, rot_angle, spacing, n_x, n_y, layers):
        if not (type(layers) == list):
            layers = [layers]
        pt1 = np.array((0, -width / 2.))
        pt2 = np.array((length, width / 2.))
        slit = Cell("Slit")
        for l in layers:
            rect = Rectangle(pt1, pt2, layer=l)
            slit.add(rect)
            shape = Cell('Branches-{}/{}/{}-lwp'.format(length, width, spacing))
            shape.add(slit, rotation=0 + rot_angle)
            shape.add(slit, rotation=120 + rot_angle)
            shape.add(slit, rotation=240 + rot_angle)

            x_spacing = length + spacing
            y_spacing = (length + spacing) * np.sin(np.deg2rad(60))
            shape_array = CellArray(shape, n_x, np.ceil(n_y / 2.), (x_spacing, y_spacing * 2.), origin=(
                -(n_x * x_spacing - spacing) / 2., -(n_y * y_spacing - spacing * np.sin(np.deg2rad(60))) / 2.))
            shape_array2 = CellArray(shape, n_x, np.ceil(n_y / 2.), (x_spacing, y_spacing * 2.), origin=(
                x_spacing / 2. - (n_x * x_spacing - spacing) / 2.,
                y_spacing - (n_y * y_spacing - spacing * np.sin(np.deg2rad(60))) / 2.))

            all_shapes = Cell('BranchArray-{}/{}/{}-lwp'.format(length, width, spacing))
            all_shapes.add(shape_array)
            all_shapes.add(shape_array2)
        return all_shapes  # %%Create the pattern that we want to write


lgField = Frame("LargeField", (2000., 2000.), [])  # Create the large write field
lgField.make_align_markers(20., 200., (850., 850.), l_lgBeam, cross=True, auto_align_marks=False)

# Define parameters that we will use for the slits
widths = [0.004, 0.008, 0.012, 0.016, 0.028, 0.044]
spacings = [1.0, 0.5]
lengths = [5., 10.]

smFrameSize = 400
slitColumnSpacing = 3.

# Create the smaller write field and corresponding markers
smField1 = Frame("SmallField1", (smFrameSize, smFrameSize), [])
# smField1.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
# smField1.make_slit_array(pitches[0], slitColumnSpacing, widths, lengths[0], rotAngle, 100, 100, 30, l_smBeam)
# smField1.make_branch_device(0.05, 1., 5., 10., 8., l_smBeam)
smField1.make_branch_device_array(spacings[0], widths, 115., 115., 30., lengths[0] - 2., lengths[0], 5, l_smBeam)

smField2 = Frame("SmallField2", (smFrameSize, smFrameSize), [])
# smField2.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField2.make_branch_device_array(spacings[0], widths, 115., 115., 30., lengths[1] - 2., lengths[1], 5, l_smBeam)

smField3 = Frame("SmallField3", (smFrameSize, smFrameSize), [])
# smField3.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField3.make_branch_device_array(spacings[1], widths, 115., 115., 30., lengths[0] - 2., lengths[0], 5, l_smBeam)

smField4 = Frame("SmallField4", (smFrameSize, smFrameSize), [])
# smField4.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField4.make_branch_device_array(spacings[1], widths, 115., 115., 30., lengths[1] - 2., lengths[1], 5, l_smBeam)

centerAlignField = Frame("CenterAlignField", (smFrameSize, smFrameSize), [])
# centerAlignField.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)

# Add everything together to a top cell
topCell = Cell("TopCell")
topCell.add(lgField)
smFrameSpacing = 400  # Spacing between the three small frames
dx = smFrameSpacing + smFrameSize
dy = smFrameSpacing + smFrameSize
topCell.add(smField1, origin=(-dx / 2., dy / 2.))
topCell.add(smField2, origin=(dx / 2., dy / 2.))
topCell.add(smField3, origin=(-dx / 2., -dy / 2.))
topCell.add(smField4, origin=(dx / 2., -dy / 2.))
topCell.add(centerAlignField, origin=(0., 0.))

# %%Create the layout and output GDS file
layout = Layout('LIBRARY')
if putOnWafer:  # Fit as many patterns on a wafer as possible
    wafer = MBEWafer('MembranesWafer', wafer_r=wafer_r, cells=[topCell], cell_gap=CELL_GAP, mkWidth=tDicingMarks,
                     cellsAtEdges=False)
    layout.add(wafer)

    # Try to poorly calculate the write time
    freq = 20E6  # 20 GHz
    spotsize = 100E-9  # 100nm beam
    gridsize = np.sqrt(2) / 2. * spotsize
    spotarea = gridsize ** 2.
    waferarea = wafer.area() / 1E6 ** 2.
    writetime = waferarea / spotarea / freq
    time = timedelta(seconds=writetime)
    print '\nEstimated write time: \n' + str(time)

# layout.show()
else:  # Only output a single copy of the pattern (not on a wafer)
    layout.add(topCell)
    layout.show()

file_string = str(waferVer) + '_' + str(density) + ' dMark' + str(tDicingMarks)
filename = file_string.replace(' ', '_') + '.gds'
layout.save(filename)
