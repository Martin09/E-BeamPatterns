# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2015

@author: Martin Friedl

wrt v1.0:
- Pitches are smaller 200, 400, 600, 800nm
- Changed holes to only include larger ones

wrt v1.2:
- Fixed pitch to only 750nm
- Added skews, which stretch the shapes in the x direction by various amounts to see if this helps
- Added vertical array of lines for Corentin's etching optimization and re-flow
- Added wafer label to the chip

wrt v1.3:
- Added alignment markers (with auto markers)
- Removed TEM membranes (used for cross sections) for faster writing
- Re-added TEM nanowire square for transfer to TEM grids
- Changed dicing lines from 10um to 8um for shorter write time
- Changed large alignment crosses from 20um width to 10um width for shorter write time
- Added export of up/down blocks for alignment jobs
everything is off by 5425.59-5471.57 in the upblocks? Write time 1h19
- Modified templates111 to shift pattern and always output it in the same position (centered on (0,0))
"""

# TODO: Finish cleaning up code warnings

import numpy as np
from shapely.affinity import rotate as rotateshape
from shapely.geometry import LineString

from GrowthTheoryCell import make_theory_cell, make_shape_array
from GrowthTheoryCell_Branches import make_theory_cell_br
from gdsCAD_v045.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_v045.shapes import Box, Rectangle, Label, Disk, RegPolygon
from gdsCAD_v045.templates111 import Wafer_TriangStyle
from gdsCAD_v045.utils import scale

WAFER_ID = '9000030923100'  # CHANGE THIS FOR EACH DIFFERENT WAFER
PATTERN = 'AN1.4'
CELL_GAP = 3000
glbAlignmentMarks = False
tDicingMarks = 8.  # Dicing mark line thickness (um)
rotAngle = 0.  # Rotation angle of the membranes
wafer_r = 25e3
waferVer = '111A NWs Wafer v1.4 r{:d}'.format(int(wafer_r / 1000))
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
        self.add_blockLabels(l_lgBeam, center=True)
        self.add_cellLabels(l_lgBeam, center=True)  # Put cell labels ('A','B','C'...) in center of each cell
        self.add_dicing_marks(l_lgBeam, mkWidth=mkWidth)  # Width of dicing marks
        self.add_sub_dicing_ticks(300, mkWidth, l_lgBeam)
        self.add_theory_cell()
        self.add_tem_membranes([0.020, 0.030, 0.040, 0.050], 1000, 1, l_smBeam)
        self.add_chip_labels()
        self.add_tem_nanowires()
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

        # for x, y in self.upCenters:
        #     self.add(tem_membranes2, origin=(x, y - 2000))
        #     self.add(tem_membranes2, origin=(x, y - 1400), rotation=90)
        # for x, y in self.downCenters:
        #     self.add(tem_membranes2, origin=(x, y + 2000), rotation=180)
        #     self.add(tem_membranes2, origin=(x, y + 1400), rotation=90)

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
            text = Label('{}'.format(lbl_dict[type.lower()]), 2, layer=layer)
            lblVertOffset = 0.8
            text.translate(
                tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                    array_size / 2., array_size / lblVertOffset))))  # Center justify label
            shape_array_cell.add(text)
        if sidelabels:
            text = Label('a={:.0f}knm2'.format(shape_area * 1000), 2, layer=layer)
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


# %%Create the pattern that we want to write
lgField = Frame("LargeField", (2000., 2000.), [])  # Create the large write field
lgField.make_align_markers(10., 200., (850., 850.), l_lgBeam, cross=True, auto_marks=True)

# Define parameters that we will use for the slits
widths = [0.004, 0.008, 0.012, 0.016, 0.028, 0.044]
pitch = 0.75
areas = [0.00125, 0.00250, 0.005, 0.010]
shapes = ['Circles', 'Tris_left', 'Tris_right']
skews = [1., 1.5, 2., 3.]

smFrameSize = 400
slitColumnSpacing = 3.
# Create the smaller write field and corresponding markers
skew = skews[0]
lbl = "Sk={:.1f}x".format(skew)
smField1 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField1.frame_label(lbl, 3, l_smBeam)
smField1.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField1.make_many_shapes(25, areas, pitch, shapes, skew, l_smBeam)

skew = skews[1]
lbl = "Sk={:.1f}x".format(skew)
smField2 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField2.frame_label(lbl, 3, l_smBeam)
smField2.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField2.make_many_shapes(25, areas, pitch, shapes, skew, l_smBeam)

skew = skews[2]
lbl = "Sk={:.1f}x".format(skew)
smField3 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField3.frame_label(lbl, 3, l_smBeam)
smField3.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField3.make_many_shapes(25, areas, pitch, shapes, skew, l_smBeam)

skew = skews[3]
lbl = "Sk={:.1f}x".format(skew)
smField4 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField4.frame_label(lbl, 3, l_smBeam)
smField4.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField4.make_many_shapes(25, areas, pitch, shapes, skew, l_smBeam)

centerAlignField = Frame("CenterAlignField", (smFrameSize, smFrameSize), [])

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
# topCell.add(centerAlignField, origin=(0., 0.))

# %%Create the layout and output GDS file
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
rect_layout.save(filename+'_etchCheck.gds')
rect_layout.add(rectCell)
# wafer.add(rectCell, origin=(0, -2000))

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
