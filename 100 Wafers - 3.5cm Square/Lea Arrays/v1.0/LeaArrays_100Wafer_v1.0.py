# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2015

@author: Martin Friedl
"""

from datetime import date

import numpy as np

from Patterns.GrowthTheoryCell_100_3BranchDevices import make_theory_cell_3br
from Patterns.GrowthTheoryCell_100_4BranchDevices import make_theory_cell_4br
from Patterns.GrowthTheoryCell_100_Large import make_theory_cell
from gdsCAD_py3.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_py3.shapes import Box, Rectangle, Label, Disk, RegPolygon
from gdsCAD_py3.templates100 import Wafer_GridStyle, dashed_line
from gdsCAD_py3.utils import scale

WAFER_ID = 'XXXX'  # CHANGE THIS FOR EACH DIFFERENT WAFER
PATTERN = 'LA1.0'
putOnWafer = True  # Output full wafer or just a single pattern?
HighDensity = False  # High density of triangles?
glbAlignmentMarks = False
tDicingMarks = 10.  # Dicing mark line thickness (um)
rotAngle = 0.  # Rotation angle of the membranes
wafer_r = 50e3
waferVer = 'Lea Arrays v1.0'.format(int(wafer_r / 1000))

waferLabel = waferVer + '\n' + date.today().strftime("%d%m%Y")
# Layers
l_smBeam = 0
l_lgBeam = 1
l_drawing = 100


# %% Wafer template for MBE growth
class MBE100Wafer(Wafer_GridStyle):
    """
    A 2" wafer divided into square cells
    """

    def __init__(self, name, cells=None):
        Wafer_GridStyle.__init__(self, name=name, cells=cells, block_gap=1200.)

        # The placement of the wafer alignment markers
        am_x = 1.5e4
        am_y = 1.5e4
        self.align_pts = np.array([am_x, am_y])
        self.align_pts = np.vstack((self.align_pts, self.align_pts *
                                    (-1, 1)))  # Reflect about y-axis
        self.align_pts = np.vstack((self.align_pts, self.align_pts *
                                    (1, -1)))  # Reflect about x-axis

        self.wafer_r = wafer_r
        self.block_size = np.array([35e3, 35e3])
        self._place_blocks(radius=self.wafer_r + 5e3)

        self.add_blocks()
        self.add_wafer_outline(layers=l_drawing)
        self.add_dashed_dicing_marks(layers=[l_lgBeam])
        self.add_block_labels(layers=[l_lgBeam])
        # self.add_prealignment_markers(layers=[l_lgBeam])
        # self.add_tem_membranes([0.08, 0.012, 0.028, 0.044], 2000, 1, l_smBeam)
        self.add_theory_cells()
        self.add_chip_labels()

        bottom = np.array([0, -self.wafer_r * 0.9])
        # top = np.array([0, -1]) * bottom
        self.add_waferLabel(waferLabel, l_drawing, pos=bottom)

    def add_block_labels(self, layers):
        txtSize = 2000
        for (i, pt) in enumerate(self.block_pts):
            origin = (pt + np.array([0.5, 0.5])) * self.block_size
            blk_lbl = self.blockcols[pt[0]] + self.blockrows[pt[1]]
            for l in layers:
                txt = Label(blk_lbl, txtSize, layer=l)
            bbox = txt.bounding_box
            offset = np.array(pt)
            txt.translate(-np.mean(bbox, 0))  # Center text around origin
            lbl_cell = Cell("lbl_" + blk_lbl)
            lbl_cell.add(txt)
            origin += np.array([0, 0])
            self.add(lbl_cell, origin=origin)

    def add_dashed_dicing_marks(self, layers):
        if type(layers) is not list:
            layers = [layers]
        width = 10. / 2
        dashlength = 2041
        r = self.wafer_r
        rng = np.floor(self.wafer_r / self.block_size).astype(int)
        dmarks = Cell('DIC_MRKS')
        for l in layers:
            for x in np.arange(-rng[0], rng[0] + 1) * self.block_size[0]:
                y = np.sqrt(r ** 2 - x ** 2)
                vm = dashed_line([x, y], [x, -y], dashlength, width, layer=l)
                dmarks.add(vm)

            for y in np.arange(-rng[1], rng[1] + 1) * self.block_size[1]:
                x = np.sqrt(r ** 2 - y ** 2)
                hm = dashed_line([x, y], [-x, y], dashlength, width, layer=l)
                dmarks.add(hm)
        self.add(dmarks)

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
            mrkr_positions = [75 * n + (n - 1) * n // 2 for n in range(1, (mrkr_size - 1) // 2 + 1)]
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

        center_x, center_y = (5000, 5000)
        for block in self.blocks:
            block.add(pamm_cell, origin=(center_x + 2000, center_y))
            block.add(pamm_cell, origin=(center_x - 2000, center_y))

    def add_tem_membranes(self, widths, length, pitch, layer):
        tem_membranes = Cell('TEM_Membranes')
        n = 5
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

        n2 = 3
        tem_membranes2 = Cell('Many_TEM_Membranes')
        tem_membranes2.add(CellArray(tem_membranes, 1, n2, (0, n * len(widths) * pitch)))

        center_x, center_y = (self.block_size[0] / 2., self.block_size[1] / 2.)
        for block in self.blocks:
            block.add(tem_membranes2, origin=(center_x, center_y + 4000))

    def add_theory_cells(self):

        theory_cells = Cell('TheoryCells')
        theory_cells.add(make_theory_cell(wafer_orient='100'), origin=(50, 0))
        # theory_cells.add(make_theory_cell_3br(), origin=(0, 0))
        # theory_cells.add(make_theory_cell_4br(), origin=(400, 0))

        center_x, center_y = (self.block_size[0] / 2., self.block_size[1] / 2.)
        for block in self.blocks:
            block.add(theory_cells, origin=(center_x, center_y - 4000))

    def add_chip_labels(self):
        wafer_lbl = PATTERN + '\n' + WAFER_ID
        text = Label(wafer_lbl, 20., layer=l_lgBeam)
        text.translate(tuple(np.array(-text.bounding_box.mean(0))))  # Center justify label
        chip_lbl_cell = Cell('chip_label')
        chip_lbl_cell.add(text)

        center_x, center_y = (self.block_size[0] / 2., self.block_size[1] / 2.)
        for block in self.blocks:
            block.add(chip_lbl_cell, origin=(center_x, center_y - 4850))


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

    def make_align_markers(self, t, w, position, layers, joy_markers=False, camps_markers=False):
        """

        :param t:
        :param w:
        :param position:
        :param layers:
        :param joy_markers:
        :param camps_markers:
        :return:
        """
        if not (type(layers) == list):
            layers = [layers]
        top_mk_cell = Cell('AlignmentMark')
        for l in layers:
            if not joy_markers:
                am0 = Rectangle((-w / 2., -w / 2.), (w / 2., w / 2.), layer=l)
                rect_mk_cell = Cell("RectMarker")
                rect_mk_cell.add(am0)
                top_mk_cell.add(rect_mk_cell)
            elif joy_markers:
                crosspts = [(0, 0), (w / 2., 0), (w / 2., t), (t, t), (t, w / 2), (0, w / 2), (0, 0)]
                crosspts.extend(tuple(map(tuple, (-np.array(crosspts)).tolist())))
                am0 = Boundary(crosspts, layer=l)  # Create gdsCAD shape
                joy_mk_cell = Cell("JOYMarker")
                joy_mk_cell.add(am0)
                top_mk_cell.add(joy_mk_cell)

            if camps_markers:
                emw = 20.  # 20 um e-beam marker width
                camps_mk = Rectangle((-emw / 2., -emw / 2.), (emw / 2., emw / 2.), layer=l)
                camps_mk_cell = Cell("CAMPSMarker")
                camps_mk_cell.add(camps_mk)
                top_mk_cell.add(camps_mk_cell, origin=[100., 100.])
                top_mk_cell.add(camps_mk_cell, origin=[100., -100.])
                top_mk_cell.add(camps_mk_cell, origin=[-100., 100.])
                top_mk_cell.add(camps_mk_cell, origin=[-100., -100.])

            self.align_markers = Cell("AlignMarkers")
            self.align_markers.add(top_mk_cell, origin=np.array(position) * np.array([1, -1]))
            self.align_markers.add(top_mk_cell, origin=np.array(position) * np.array([-1, -1]))
            self.align_markers.add(top_mk_cell, origin=np.array(position) * np.array([1, 1]))
            self.align_markers.add(top_mk_cell, origin=np.array(position) * np.array([-1, 1]))
            self.add(self.align_markers)

    # TODO: Center array around the origin
    def make_shape_array(self, array_size, shape_area, shape_pitch, type, layer, skew, toplabels=False,
                         sidelabels=False):
        """

        :param array_size:
        :param shape_area:
        :param shape_pitch:
        :param type:
        :param layer:
        :param skew:
        :param toplabels:
        :param sidelabels:
        :return:
        """
        num_of_shapes = int(np.ceil(array_size / shape_pitch))
        base_cell = Cell('Base')

        if 'tri' in type.lower():
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
        elif type.lower() == "circle":
            circ_radius = np.sqrt(shape_area / np.pi)
            circ = scale(Disk([0, 0], circ_radius, layer=layer), [skew, 1.0])
            base_cell.add(circ)
        elif type.lower() == 'hexagon':
            hex_side = np.sqrt(shape_area / 6. / np.sqrt(3) * 4)
            hex_shape = scale(RegPolygon([0, 0], hex_side, 6, layer=layer), [skew, 1.0])
            hex_cell = Cell('Hex')
            hex_cell.add(hex_shape)
            base_cell.add(hex_cell, rotation=0)
        elif type.lower() == 'square':
            sq_side = np.sqrt(shape_area)
            sq_shape = scale(Rectangle([-sq_side / 2., -sq_side / 2.], [sq_side / 2., sq_side / 2.], layer=layer),
                             [skew, 1.0])
            sq_cell = Cell('Square')
            sq_cell.add(sq_shape)
            base_cell.add(sq_cell, rotation=0)

        shape_array = CellArray(base_cell, num_of_shapes, num_of_shapes, [shape_pitch, shape_pitch])
        shape_array_cell = Cell('Shape Array')
        shape_array_cell.add(shape_array)

        lbl_dict = {'hexagon': 'hex', 'circle': 'circ', 'tris_right': 'triR', 'tris_left': 'triL', 'square': 'sqr'}

        if toplabels:
            text = Label('p={:.0f}nm'.format(shape_pitch*1000), 10., layer=l_lgBeam)
            lblVertOffset = 0.8
            text.translate(
                tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                    array_size / 2., array_size / lblVertOffset))))  # Center justify label
            shape_array_cell.add(text)
        if sidelabels:
            text = Label('s={:.0f}nm'.format(np.sqrt(shape_area) * 1000), 10., layer=l_lgBeam)
            lblHorizOffset = 1.5
            text.translate(
                tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                    -array_size / lblHorizOffset, array_size / 2.))))  # Center justify label
            shape_array_cell.add(text)

        return shape_array_cell

    def make_many_shapes(self, array_size, shape_areas, pitches, shape, skew, layer):
        """

        :param array_size:
        :param shape_areas:
        :param pitches:
        :param shape:
        :param skew:
        :param layer:
        :return:
        """
        if (type(shape) == list):
            shape = shape[0]
        offset_x = array_size * 1.25
        offset_y = array_size * 1.25
        cur_y = 0
        many_shape_cell = Cell('ManyShapes')
        for area in shape_areas:
            cur_x = 0
            for pitch in pitches:
                write_top_labels = cur_y == 0
                write_side_labels = cur_x == 0
                s_array = self.make_shape_array(array_size, area, pitch, shape, layer, skew, toplabels=write_top_labels,
                                                sidelabels=write_side_labels)
                many_shape_cell.add(s_array, origin=(cur_x - array_size / 2., cur_y - array_size / 2.))
                cur_x += offset_x
            cur_y -= offset_y
        self.add(many_shape_cell, origin=(-offset_x * (len(pitches) - 1) / 2., offset_y * (len(shape_areas) - 1) / 2.))


# %%Create the pattern that we want to write

lgField = Frame("LargeField", (2000., 2000.), [])  # Create the large write field
lgField.make_align_markers(20., 200., (850., 850.), l_lgBeam, joy_markers=True, camps_markers=True)

# Define parameters that we will use for the shapes
pitches = [0.5, 1.0, 1.5]
areas = [0.050 ** 2, 0.100 ** 2, 0.250 ** 2]
shape = ['Square']
skew = 1
smFrameSize = 400
slitColumnSpacing = 3.
array_size = 100

centerAlignField = Frame("CenterAlignField", (smFrameSize, smFrameSize), [])
# centerAlignField.make_align_markers(2., 20., (180., 180.), l_lgBeam, joy_markers=True)
centerAlignField.make_many_shapes(array_size, areas, pitches, shape, skew, l_smBeam)

# Add everything together to a top cell
topCell = Cell("TopCell")
topCell.add(lgField)
smFrameSpacing = 400  # Spacing between the three small frames
dx = smFrameSpacing + smFrameSize
dy = smFrameSpacing + smFrameSize
# topCell.add(smField1, origin=(-dx / 2., dy / 2.))
# topCell.add(smField1, origin=(dx / 2., dy / 2.))
# topCell.add(smField1, origin=(-dx / 2., -dy / 2.))
# topCell.add(smField1, origin=(dx / 2., -dy / 2.))
topCell.add(centerAlignField, origin=(0., 0.))
topCell.spacing = np.array([12000., 12000.])

# %%Create the layout and output GDS file
layout = Layout('LIBRARY')
if putOnWafer:  # Fit as many patterns on a 2inch wafer as possible
    wafer = MBE100Wafer('MembranesWafer', cells=[topCell])
    layout.add(wafer)
# layout.show()
else:  # Only output a single copy of the pattern (not on a wafer)
    layout.add(topCell)
    layout.show()

filestring = str(waferVer) + '_' + WAFER_ID + '_' + date.today().strftime("%d%m%Y") + ' dMark' + str(tDicingMarks)
filename = filestring.replace(' ', '_') + '.gds'
layout.save(filename)

cell_layout = Layout('LIBRARY')
cell_layout.add(wafer.blocks[0])
cell_layout.save(filestring.replace(' ', '_') + '_block' + '.gds')
