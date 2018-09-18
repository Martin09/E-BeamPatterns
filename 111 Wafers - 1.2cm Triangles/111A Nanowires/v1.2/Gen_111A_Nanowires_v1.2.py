# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2015

@author: Martin Friedl

wrt v1.0:
- Pitches are smaller 200, 400, 600, 800nm
- Changed holes to only include larger ones

"""

# TODO: Finish cleaning up code warnings

from datetime import timedelta

import numpy as np
from gdsCAD_v045.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_v045.shapes import Box, Rectangle, Label, Disk, RegPolygon
from shapely.affinity import rotate as rotateshape
from shapely.geometry import LineString

from GrowthTheoryCell import make_theory_cell, make_shape_array
from GrowthTheoryCell_Branches import make_theory_cell_br
from gdsCAD_v045.templates111 import Wafer_TriangStyle

CELL_GAP = 3000
glbAlignmentMarks = False
tDicingMarks = 10.  # Dicing mark line thickness (um)
rotAngle = 0.  # Rotation angle of the membranes
wafer_r = 25e3
waferVer = '111A NWs Wafer v1.2 r{:d}'.format(int(wafer_r / 1000))
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
        self.add_sub_dicing_ticks(300, 10, l_lgBeam)
        self.add_theory_cell()
        self.add_tem_membranes([0.08, 0.012, 0.028, 0.044], 2000, 1, l_smBeam)
        self.add_tem_nanowires()
        bottom = np.array([0, -self.wafer_r * 0.90])
        # Write label on layer 100 to avoid writing (and saving writing time)
        self.add_waferLabel(waferLabel, 100, pos=bottom)

    def add_theory_cell(self):

        theory_cells = Cell('TheoryCells')
        theory_cells.add(make_theory_cell(), origin=(-200, 0))
        theory_cells.add(make_theory_cell_br(), origin=(200, 0))

        for x, y in self.upCenters:
            self.add(theory_cells, origin=(x, y + 1300))
        for x, y in self.downCenters:
            self.add(theory_cells,
                     origin=(x, y - 1300))  # Don't rotate because of directionality of branched membranes

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

        for x, y in self.upCenters:
            self.add(tem_membranes2, origin=(x, y - 2000))
        for x, y in self.downCenters:
            self.add(tem_membranes2, origin=(x, y + 2000), rotation=180)

    def add_tem_nanowires(self):
        size = 500
        y_offset = 1300
        shapes_big = make_shape_array(size, 0.02, 0.5, 'Tris_right', l_smBeam, labels=False)
        shapes_small = make_shape_array(size, 0.005, 0.5, 'Tris_right', l_smBeam, labels=False)
        tem_shapes = Cell('TEMShapes')
        # tem_shapes.add(shapes_big, origin=(2200 - size / 2., y_offset - size / 2.))
        tem_shapes.add(shapes_small, origin=(-2200 - size / 2., y_offset - size / 2.))

        for x, y in self.upCenters:
            self.add(tem_shapes, origin=(x, y))
        for x, y in self.downCenters:
            self.add(tem_shapes, origin=(x, y - 2 * y_offset))  # Don't rotate because of directionality

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

    def make_align_markers(self, t, w, position, layers, cross=False):
        if not (type(layers) == list):
            layers = [layers]
        self.align_markers = Cell("AlignMarkers")
        for l in layers:
            if not cross:
                am0 = Rectangle((-w / 2., -w / 2.), (w / 2., w / 2.), layer=l)
            elif cross:
                crosspts = [(0, 0), (w / 2., 0), (w / 2., t), (t, t), (t, w / 2), (0, w / 2), (0, 0)]
                crosspts.extend(tuple(map(tuple, (-np.array(crosspts)).tolist())))

                #                crosspts = [(-t / 2., t / 2.), (-t / 2., h / 2.), (t / 2., h / 2.),
                #                            (t / 2., t / 2.), (w / 2., t / 2.), (w / 2., -t / 2.),
                #                            (t / 2., -t / 2.), (t / 2., -h / 2.),
                #                            (-t / 2., -h / 2.), (-t / 2., -t / 2.),
                #                            (-w / 2., -t / 2.), (-w / 2., t / 2.)]
                am0 = Boundary(crosspts, layer=l)  # Create gdsCAD shape
                # am1 = Polygon(crosspts) #Create shapely polygon for later calculation

            am1 = am0.copy().translate(tuple(np.array(position) * [1, 1]))  # 850,850
            am2 = am0.copy().translate(tuple(np.array(position) * [-1, 1]))  # 850,850
            am3 = am0.copy().translate(tuple(np.array(position) * [1, -1]))  # 850,850
            am4 = am0.copy().translate(tuple(np.array(position) * [-1, -1]))  # 850,850
            #            am4 = am0.copy().scale((-1, -1))  #Reflect in both x and y-axis
            self.align_markers.add([am1, am2, am3, am4])
            self.add(self.align_markers)

    # TODO: Center array around the origin
    def make_shape_array(self, array_size, shape_area, shape_pitch, type, layer, toplabels=False, sidelabels=False):
        num_of_shapes = int(np.ceil(array_size / shape_pitch))
        base_cell = Cell('Base')

        if 'tris' in type.lower():
            triangle_side = np.sqrt(shape_area / np.sqrt(3) * 4)
            tri_shape = RegPolygon([0, 0], triangle_side, 3, layer=layer)
            tri_cell = Cell('Tri')
            tri_cell.add(tri_shape)
            if 'right' in type.lower():
                base_cell.add(tri_cell, rotation=0)
            elif 'left' in type.lower():
                base_cell.add(tri_cell, rotation=60)
            elif 'down' in type.lower():
                base_cell.add(tri_cell, rotation=30)
            elif 'up' in type.lower():
                base_cell.add(tri_cell, rotation=-30)
        elif type.lower() == "circles":
            circ_radius = np.sqrt(shape_area / np.pi)
            circ = Disk([0, 0], circ_radius, layer=layer)
            base_cell.add(circ)
        elif type.lower() == 'hexagons':
            hex_side = np.sqrt(shape_area / 6. / np.sqrt(3) * 4)
            hex_shape = RegPolygon([0, 0], hex_side, 6, layer=layer)
            hex_cell = Cell('Hex')
            hex_cell.add(hex_shape)
            base_cell.add(hex_cell, rotation=0)

        shape_array = CellArray(base_cell, num_of_shapes, num_of_shapes, [shape_pitch, shape_pitch])
        shape_array_cell = Cell('Shape Array')
        shape_array_cell.add(shape_array)

        if toplabels:
            text = Label('{}'.format(type), 2)
            lblVertOffset = 0.8
            text.translate(
                tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                    array_size / 2., array_size / lblVertOffset))))  # Center justify label
            shape_array_cell.add(text)
        if sidelabels:
            text = Label('a={:.0f}nm2'.format(shape_area * 1000 ** 2), 2)
            lblHorizOffset = 1.5
            text.translate(
                tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                    -array_size / lblHorizOffset, array_size / 2.))))  # Center justify label
            shape_array_cell.add(text)

        return shape_array_cell

    def make_many_shapes(self, array_size, shape_areas, pitch, shapes, layer):
        offset_x = array_size * 1.25
        offset_y = array_size * 1.25
        cur_y = 0
        many_shape_cell = Cell('ManyShapes')
        for area in shape_areas:
            cur_x = 0
            for shape in shapes:
                write_top_labels = cur_y == 0
                write_side_labels = cur_x == 0
                s_array = self.make_shape_array(array_size, area, pitch, shape, layer, toplabels=write_top_labels,
                                                sidelabels=write_side_labels)
                many_shape_cell.add(s_array, origin=(cur_x - array_size / 2., cur_y - array_size / 2.))
                cur_x += offset_x
            cur_y -= offset_y
        self.add(many_shape_cell, origin=(-offset_x * (len(shapes) - 1) / 2., offset_y * (len(pitches) - 1) / 2.))


# %%Create the pattern that we want to write
lgField = Frame("LargeField", (2000., 2000.), [])  # Create the large write field
lgField.make_align_markers(20., 200., (850., 850.), l_lgBeam, cross=True)

# Define parameters that we will use for the slits
widths = [0.004, 0.008, 0.012, 0.016, 0.028, 0.044]
pitches = [0.2, 0.4, 0.6, 0.8]
areas = [0.00125, 0.00250, 0.005, 0.010]
shapes = ['Hexagons', 'Circles', 'Tris_left', 'Tris_right']

smFrameSize = 400
slitColumnSpacing = 3.

# Create the smaller write field and corresponding markers
pitch = pitches[0]
lbl = "P={:.0f}nm".format(pitch * 1000)
smField1 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField1.frame_label(lbl, 5, l_lgBeam)
smField1.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField1.make_many_shapes(25, areas, pitch, shapes, l_smBeam)

pitch = pitches[1]
lbl = "P={:.0f}nm".format(pitch * 1000)
smField2 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField2.frame_label(lbl, 5, l_lgBeam)
smField2.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField2.make_many_shapes(25, areas, pitch, shapes, l_smBeam)

pitch = pitches[2]
lbl = "P={:.0f}nm".format(pitch * 1000)
smField3 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField3.frame_label(lbl, 5, l_lgBeam)
smField3.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField3.make_many_shapes(25, areas, pitch, shapes, l_smBeam)

pitch = pitches[3]
lbl = "P={:.0f}nm".format(pitch * 1000)
smField4 = Frame(lbl, (smFrameSize, smFrameSize), [])
smField4.frame_label(lbl, 5, l_lgBeam)
smField4.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField4.make_many_shapes(25, areas, pitch, shapes, l_smBeam)

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
topCell.add(centerAlignField, origin=(0., 0.))

# %%Create the layout and output GDS file
layout = Layout('LIBRARY', precision=1e-10)

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
print(('\nEstimated write time: \n' + str(time)))

file_string = str(waferVer)
filename = file_string.replace(' ', '_') + '.gds'
layout.save(filename)
