# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2015

@author: Martin Friedl

TODO: Add the branched membrane devices!
TODO: Add serial number of wafers that I will expose

"""

# TODO: Finish cleaning up code warnings

import numpy as np
from shapely.affinity import rotate as rotateshape
from shapely.geometry import LineString

from GrowthTheoryCell import make_theory_cell, make_shape_array
from GrowthTheoryCell_Branches import make_theory_cell_br
from QuantumPlayground_v1 import make_qp
from gdsCAD_v045.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_v045.shapes import Box, Rectangle, Label, Disk, RegPolygon
from gdsCAD_v045.templates111 import Wafer_TriangStyle
from gdsCAD_v045.utils import scale

WAFER_ID = 'XXXXXXXXXXX'  # CHANGE THIS FOR EACH DIFFERENT WAFER
PATTERN = 'BM4.9'
CELL_GAP = 3000
glbAlignmentMarks = False
tDicingMarks = 8.  # Dicing mark line thickness (um)
rotAngle = 0.  # Rotation angle of the membranes
wafer_r = 25e3
waferVer = 'Basel Membranes v4.8.1 r{:d}'.format(int(wafer_r / 1000))
waferLabel = waferVer
mkWidthMinor = 3  # Width of minor (Basel) markers within each triangular chip

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
                 wafer_r=25e3,
                 trisize=12e3,
                 cellsize=2e3,
                 block_gap=0.,  # 1200
                 cell_gap=800.,  # 800
                 doMCSearch=True,
                 MCIterations=100,
                 doMCBlockSearch=True,
                 MCBlockIterations=100,
                 mkWidth=10.,
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
        # points = self.add_dicing_marks(l_lgBeam, mkWidth=mkWidth)  # Width of dicing marks
        # self.make_basel_align_marks(points, l_lgBeam, mk_width=mkWidthMinor)
        self.add_wafer_outline(100)
        self.build_and_add_blocks()
        self.add_blockLabels(l_lgBeam, center=True, quasi_unique_labels=True)
        self.add_cellLabels(l_lgBeam, center=True)  # Put cell labels ('A','B','C'...) in center of each cell
        self.add_dicing_marks(l_lgBeam, mkWidth=mkWidth)  # Width of dicing marks
        self.add_sub_dicing_ticks(300, mkWidth, l_lgBeam)
        self.add_theory_cell()
        self.add_tem_membranes([0.020, 0.030, 0.040, 0.050], 1000, 1, l_smBeam)
        self.add_chip_labels()
        self.add_prealignment_markers(l_lgBeam)
        # self.add_tem_nanowires()
        bottom = np.array([0, -self.wafer_r * 0.90])
        # Write label on layer 100 to avoid writing (and saving writing time)
        self.add_waferLabel(waferLabel, 100, pos=bottom)

    def add_chip_labels(self):
        wafer_lbl = PATTERN + '\n' + WAFER_ID
        text = Label(wafer_lbl, 10., layer=l_lgBeam)
        text.translate(tuple(np.array(-text.bounding_box.mean(0))))  # Center justify label
        chip_lbl_cell = Cell('chip_label')
        chip_lbl_cell.add(text)

        self.block_up.add(chip_lbl_cell, origin=(0, -3000))
        self.block_down.add(chip_lbl_cell, origin=(0, 3000))

    def add_theory_cell(self):

        theory_cells = Cell('TheoryCells')
        theory_cells.add(make_theory_cell(), origin=(-200, 0))
        theory_cells.add(make_theory_cell_br(), origin=(200, 0))

        self.block_up.add(theory_cells, origin=(0, 1300))
        self.block_down.add(theory_cells, origin=(0, -1300))

    def add_prealignment_markers(self, layers, mrkr_size=7):
        if mrkr_size % 2 == 0:  # Number is even, but we need odd numbers
            mrkr_size += 1
        if type(layers) is not list:
            layers = [layers]

        for l in layers:
            rect_size = 10.  # 10 um large PAMM rectangles
            marker_rect = Rectangle([-rect_size / 2., -rect_size / 2.], [rect_size / 2., rect_size / 2.], layer=l)
            marker = Cell('10umMarker')
            marker.add(marker_rect)

            # Make one arm of the PAMM array
            marker_arm = Cell('PAMM_Arm')
            # Define the positions of the markers, they increase in spacing by 1 um each time:
            mrkr_positions = [75 * n + (n - 1) * n / 2 for n in xrange(1, (mrkr_size - 1) / 2 + 1)]
            for pos in mrkr_positions:
                marker_arm.add(marker, origin=[pos, 0])

            # Build the final PAMM Marker
            pamm_cell = Cell('PAMM_Marker')
            pamm_cell.add(marker)  # Center marker
            pamm_cell.add(marker_arm)  # Right arm
            pamm_cell.add(marker_arm, rotation=180)  # Left arm
            pamm_cell.add(marker_arm, rotation=90)  # Top arm
            pamm_cell.add(marker_arm, rotation=-90)  # Bottom arm
            for pos in mrkr_positions:
                pamm_cell.add(marker_arm, origin=[pos, 0], rotation=90)  # Top arms
                pamm_cell.add(marker_arm, origin=[-pos, 0], rotation=90)
                pamm_cell.add(marker_arm, origin=[pos, 0], rotation=-90)  # Bottom arms
                pamm_cell.add(marker_arm, origin=[-pos, 0], rotation=-90)

            # Make the 4 tick marks that mark the center of the array
            h = 30.
            w = 100.
            tick_mrk = Rectangle([-w / 2., -h / 2.], [w / 2, h / 2.], layer=l)
            tick_mrk_cell = Cell("TickMark")
            tick_mrk_cell.add(tick_mrk)
            pos = mrkr_positions[-1] + 75 + w / 2.
            pamm_cell.add(tick_mrk_cell, origin=[pos, 0])
            pamm_cell.add(tick_mrk_cell, origin=[-pos, 0])
            pamm_cell.add(tick_mrk_cell, origin=[0, pos], rotation=90)
            pamm_cell.add(tick_mrk_cell, origin=[0, -pos], rotation=90)

        self.block_up.add(pamm_cell, origin=(2000, 500))
        self.block_up.add(pamm_cell, origin=(-2000, 500))
        self.block_down.add(pamm_cell, origin=(2000, -500))
        self.block_down.add(pamm_cell, origin=(-2000, -500))

    def add_tem_membranes(self, widths, length, pitch, layer):

        tem_membranes = Cell('TEM_Membranes')

        n = 3
        curr_y = 0
        for width in widths:
            membrane = Path([(-length / 2., 0), (length / 2., 0)], width=width, layer=layer)
            membrane_cell = Cell('Membrane_w{:.0f}'.format(width * 1000))
            membrane_cell.add(membrane)
            membrane_array = CellArray(membrane_cell, 1, n, (0, pitch))
            membrane_array_cell = Cell('MembraneArray_w{:.0f}'.format(width * 1000))
            membrane_array_cell.add(membrane_array)
            tem_membranes.add(membrane_array_cell, origin=(0, curr_y))
            curr_y += n * pitch

        n2 = 5
        tem_membranes2 = Cell('Many_TEM_Membranes')
        tem_membranes2.add(CellArray(tem_membranes, 1, n2, (0, n * len(widths) * pitch)))

        self.block_up.add(tem_membranes2, origin=(0, -2000))
        # self.block_up.add(tem_membranes2, origin=(0, -1400), rotation=90)

        self.block_down.add(tem_membranes2, origin=(0, 2000))
        # self.block_down.add(tem_membranes2, origin=(0, 1400), rotation=90)

    def add_tem_nanowires(self):
        size = 500
        y_offset = 1000
        shapes_big = make_shape_array(size, 0.02, 0.5, 'Tris_right', l_smBeam, labels=False)
        shapes_small = make_shape_array(size, 0.005, 0.5, 'Tris_right', l_smBeam, labels=False)
        tem_shapes = Cell('TEMShapes')
        # tem_shapes.add(shapes_big, origin=(2200 - size / 2., y_offset - size / 2.))
        tem_shapes.add(shapes_small, origin=(-size / 2., -size / 2.))

        self.block_up.add(tem_shapes, origin=(-2200, y_offset))
        self.block_down.add(tem_shapes, origin=(2200, -y_offset))

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

    def frame_label(self, label_txt, size, beam):
        text = Label(label_txt, size, layer=beam)
        lblVertOffset = 0.4
        text.translate(
            tuple(np.array(-text.bounding_box.mean(0))))  # Center justify label
        lbl_cell = Cell('frame_label')
        lbl_cell.add(text)
        self.add(lbl_cell, origin=(0, self.size_y * lblVertOffset / 2.))

    def make_align_markers(self, t, w, position, layers, cross=False, auto_marks=False):
        if not (type(layers) == list):
            layers = [layers]
        self.align_markers = Cell("AlignMarkers")
        self.align_marker = Cell("AlignMarker")
        for l in layers:
            if not cross:
                am0 = Rectangle((-w / 2., -w / 2.), (w / 2., w / 2.), layer=l)
                self.align_marker.add(am0)
            elif cross:
                crosspts = [(0, 0), (w / 2., 0), (w / 2., t), (t, t), (t, w / 2), (0, w / 2), (0, 0)]
                crosspts.extend(tuple(map(tuple, (-np.array(crosspts)).tolist())))

                #                crosspts = [(-t / 2., t / 2.), (-t / 2., h / 2.), (t / 2., h / 2.),
                #                            (t / 2., t / 2.), (w / 2., t / 2.), (w / 2., -t / 2.),
                #                            (t / 2., -t / 2.), (t / 2., -h / 2.),
                #                            (-t / 2., -h / 2.), (-t / 2., -t / 2.),
                #                            (-w / 2., -t / 2.), (-w / 2., t / 2.)]
                am0 = Boundary(crosspts, layer=l)  # Create gdsCAD shape
                self.align_marker.add(am0)
                # am1 = Polygon(crosspts) #Create shapely polygon for later calculation

            if auto_marks:  # automatic alignment marks for the e-beam tool
                auto_mark_rect = Rectangle((-10., -10.), (10., 10.), layer=l)
                auto_mark = Cell("AutoMark")
                auto_mark.add(auto_mark_rect)
                self.align_marker.add(auto_mark, origin=(100, 100))
                self.align_marker.add(auto_mark, origin=(-100, 100))
                self.align_marker.add(auto_mark, origin=(100, -100))
                self.align_marker.add(auto_mark, origin=(-100, -100))

            self.align_markers.add(self.align_marker, origin=tuple(np.array(position) * [1, 1]))
            self.align_markers.add(self.align_marker, origin=tuple(np.array(position) * [-1, 1]))
            self.align_markers.add(self.align_marker, origin=tuple(np.array(position) * [1, -1]))
            self.align_markers.add(self.align_marker, origin=tuple(np.array(position) * [-1, -1]))
            self.add(self.align_markers)

    # TODO: Center array around the origin
    def make_shape_array(self, array_size, shape_area, shape_pitch, type, layer, skew, toplabels=False,
                         sidelabels=False):
        num_of_shapes = int(np.ceil(array_size / shape_pitch))
        base_cell = Cell('Base')

        if 'tris' in type.lower():
            triangle_side = np.sqrt(shape_area / np.sqrt(3) * 4)
            tri_shape = scale(RegPolygon([0, 0], triangle_side, 3, layer=layer), [skew, 1.0])
            tri_cell = Cell('Tri')
            tri_cell.add(tri_shape)
            if 'right' in type.lower():
                base_cell.add(tri_cell, rotation=0)
            elif 'left' in type.lower():
                base_cell.add(tri_cell, rotation=-180)
            elif 'down' in type.lower():
                base_cell.add(tri_cell, rotation=30)  # not working for skew yet
            elif 'up' in type.lower():
                base_cell.add(tri_cell, rotation=-30)  # not working for skew yet
        elif type.lower() == "circles":
            circ_radius = np.sqrt(shape_area / np.pi)
            circ = scale(Disk([0, 0], circ_radius, layer=layer), [skew, 1.0])
            base_cell.add(circ)
        elif type.lower() == 'hexagons':
            hex_side = np.sqrt(shape_area / 6. / np.sqrt(3) * 4)
            hex_shape = scale(RegPolygon([0, 0], hex_side, 6, layer=layer), [skew, 1.0])
            hex_cell = Cell('Hex')
            hex_cell.add(hex_shape)
            base_cell.add(hex_cell, rotation=0)

        shape_array = CellArray(base_cell, num_of_shapes, num_of_shapes, [shape_pitch, shape_pitch])
        shape_array_cell = Cell('Shape Array')
        shape_array_cell.add(shape_array)

        lbl_dict = {'hexagons': 'hex', 'circles': 'circ', 'tris_right': 'triR', 'tris_left': 'triL'}

        if toplabels:
            text = Label('{}'.format(lbl_dict[type.lower()]), 2, layer=l_smBeam)
            lblVertOffset = 0.8
            text.translate(
                tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                    array_size / 2., array_size / lblVertOffset))))  # Center justify label
            shape_array_cell.add(text)
        if sidelabels:
            text = Label('a={:.0f}knm2'.format(shape_area * 1000), 2, layer=l_smBeam)
            lblHorizOffset = 1.5
            text.translate(
                tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                    -array_size / lblHorizOffset, array_size / 2.))))  # Center justify label
            shape_array_cell.add(text)

        return shape_array_cell

    def make_many_shapes(self, array_size, shape_areas, pitch, shapes, skew, layer):
        offset_x = array_size * 1.25
        offset_y = array_size * 1.25
        cur_y = 0
        many_shape_cell = Cell('ManyShapes')
        for area in shape_areas:
            cur_x = 0
            for shape in shapes:
                write_top_labels = cur_y == 0
                write_side_labels = cur_x == 0
                s_array = self.make_shape_array(array_size, area, pitch, shape, layer, skew, toplabels=write_top_labels,
                                                sidelabels=write_side_labels)
                many_shape_cell.add(s_array, origin=(cur_x - array_size / 2., cur_y - array_size / 2.))
                cur_x += offset_x
            cur_y -= offset_y
        self.add(many_shape_cell, origin=(-offset_x * (len(shapes) - 1) / 2., offset_y * (len(skews) - 1) / 2.))

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
                text = Label('w/s/l\n%i/%.1f/%i' % (width * 1000, spacing, len_outer), 5, layer=l_smBeam)
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

    def make_slit_array(self, _pitches, spacing, _widths, _lengths, rot_angle,
                        array_height, array_width, array_spacing, layers):
        if not (type(layers) == list):
            layers = [layers]
        if not (type(_pitches) == list):
            _pitches = [_pitches]
        if not (type(_lengths) == list):
            _lengths = [_lengths]
        if not (type(_widths) == list):
            _widths = [_widths]
        manyslits = i = j = None
        for l in layers:
            i = -1
            j = -1
            manyslits = Cell("SlitArray")
            pitch = _pitches[0]
            for length in _lengths:
                j += 1
                i = -1

                for width in _widths:
                    # for pitch in pitches:
                    i += 1
                    if i % 3 == 0:
                        j += 1  # Move to array to next line
                        i = 0  # Restart at left

                    pitch_v = pitch / np.cos(np.deg2rad(rot_angle))
                    #                    widthV = width / np.cos(np.deg2rad(rotAngle))
                    nx = int(array_width / (length + spacing))
                    ny = int(array_height / pitch_v)
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
                    text = Label('w/p/l\n%i/%i/%i' %
                                 (width * 1000, pitch, length), 5, layer=l_smBeam)
                    lbl_vertical_offset = 1.35
                    if j % 2 == 0:
                        text.translate(
                            tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                                0, -array_height / lbl_vertical_offset))))  # Center justify label
                    else:
                        text.translate(
                            tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                                0, array_height / lbl_vertical_offset))))  # Center justify label
                    slit_array.add(text)
                    manyslits.add(slit_array,
                                  origin=((array_width + array_spacing) * i, (
                                          array_height + 2. * array_spacing) * j - array_spacing / 2.))

        self.add(manyslits,
                 origin=(-i * (array_width + array_spacing) / 2, -(j + 1.5) * (
                         array_height + array_spacing) / 2))


# %%Create the pattern that we want to write

lgField = Frame("LargeField", (2000., 2000.), [])  # Create the large write field
lgField.make_align_markers(10., 200., (850., 850.), l_lgBeam, cross=True, auto_marks=True)

# Define parameters that we will use for the slits
widths = [0.016, 0.028, 0.044, 0.016, 0.028, 0.044]
pitches = [1.0, 2.0]
lengths = [10., 20.]

smFrameSize = 400
smFrameSpacing = 400  # Spacing between the three small frames
dx = smFrameSpacing + smFrameSize
dy = smFrameSpacing + smFrameSize
slitColumnSpacing = 3.

# Create the smaller write field and corresponding markers
smField1 = Frame("SmallField1", (smFrameSize, smFrameSize), [])
smField1.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField1.make_slit_array(pitches[0], slitColumnSpacing, widths, lengths[1], rotAngle, 100, 100, 30, l_smBeam)
lgField.add(smField1, origin=(-dx / 2., dy / 2.))

smField2 = Frame("SmallField2", (smFrameSize, smFrameSize), [])
smField2.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField2.make_slit_array(pitches[0], slitColumnSpacing, widths, lengths[1], rotAngle, 100, 100, 30, l_smBeam)
lgField.add(smField2, origin=(dx / 2., dy / 2.))

smField3 = Frame("SmallField3", (smFrameSize, smFrameSize), [])
smField3.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField3.make_branch_device_array(pitches[0], widths, 115., 115., 30., lengths[1] - 2., lengths[1], 5, l_smBeam)
lgField.add(smField3, origin=(-dx / 2., -dy / 2.))

smField4 = Frame("SmallField4", (smFrameSize, smFrameSize), [])
smField4.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField4.make_branch_device_array(pitches[0], widths, 115., 115., 30., lengths[1] - 2., lengths[1], 5, l_smBeam)
lgField.add(smField4, origin=(dx / 2., -dy / 2.))

centerAlignField = Frame("CenterAlignField", (smFrameSize, smFrameSize), [])
centerAlignField.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)

qp_cell = make_qp()
centerAlignField.add(qp_cell, origin=(0, 0))
lgField.add(centerAlignField, origin=(dx / 2., 0.))

emptyAlignField = Frame("EmptyAlignField", (smFrameSize, smFrameSize), [])
emptyAlignField.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
lgField.add(emptyAlignField, origin=(-dx / 2., 0))
lgField.add(emptyAlignField, origin=(0, dy / 2.))
lgField.add(emptyAlignField, origin=(0, -dy / 2.))

topCell = Cell("TopCell")
topCell.add(lgField)

# Create the layout and output GDS file
layout = Layout('LIBRARY', precision=1e-10)

wafer = MBEWafer('MembranesWafer', wafer_r=wafer_r, cells=[topCell], cell_gap=CELL_GAP, mkWidth=tDicingMarks,
                 cellsAtEdges=False)
file_string = str(waferVer)
filename = file_string.replace(' ', '_')

# Add pattern for ellipsometry check of SiO2 etching
size = 2000
rect = Rectangle((size / 2., size / 2.), (-size / 2., -size / 2.), layer=10)
rectCell = Cell('EtchCheckSquare')
rectCell.add(rect)
rect_layout = Layout('LIBRARY')
rect_layout.add(rectCell)
rect_layout.save(filename + '_etchCheck.gds')
rect_layout.add(rectCell)

layout.add(wafer)
layout.save(filename + '.gds')

# Output down chip for doing aligned jobs
layout_down = Layout('LIBRARY')
layout_down.add(wafer.block_down)
layout_down.save(filename + '_downblock.gds')

# Output up chip for doing aligned jobs
layout_up = Layout('LIBRARY')
layout_up.add(wafer.block_up)
layout_up.save(filename + '_upblock.gds')
