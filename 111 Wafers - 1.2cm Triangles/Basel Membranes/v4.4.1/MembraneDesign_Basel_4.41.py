# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2015

@author: Martin Friedl
"""

# TODO: Finish cleaning up code warnings
# TODO: Transpose the labelling. Have each row correspond to a letter, and each position in the row correspond to a number

from datetime import timedelta, date

import numpy as np
from gdsCAD_py3.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_py3.shapes import Box, Rectangle, Label
from shapely.affinity import rotate as rotateshape
from shapely.geometry import LineString

from Patterns.GrowthTheoryCell import make_theory_cell, make_shape_array
from Patterns.GrowthTheoryCell_Branches import make_theory_cell_br
from gdsCAD_py3.templates111 import Wafer_TriangStyle, dashed_line

putOnWafer = True  # Output full wafer or just a single pattern?
HighDensity = False  # High density of triangles?
glbAlignmentMarks = False
tDicingMarks = 10.  # Dicing mark line thickness (um)
rotAngle = 0.  # Rotation angle of the membranes
wafer_r = 25e3
waferVer = 'Membranes Wafer v4.41 r{:d}'.format(int(wafer_r / 1000))
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
        self.add_wafer_outline(100)
        self.build_and_add_blocks()
        self.add_blockLabels(l_lgBeam, center=True)
        self.add_cellLabels(l_lgBeam, center=True)  # Put cell labels ('A','B','C'...) in center of each cell
        self.add_dicing_marks(l_lgBeam, mkWidth=mkWidth)  # Width of dicing marks
        self.add_sub_dicing_ticks(300, mkWidth, l_lgBeam)
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
        shapes_big = make_shape_array(size, 0.02, 0.75, 'hexagons', l_smBeam, labels=False)
        shapes_small = make_shape_array(size, 0.005, 0.75, 'hexagons', l_smBeam, labels=False)  # Changed this wrt BM4.4
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
                                 (width * 1000, pitch, length), 5)
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
lgField.make_align_markers(20., 200., (850., 850.), l_lgBeam, cross=True)

# Define parameters that we will use for the slits
widths = [0.004, 0.008, 0.012, 0.016, 0.028, 0.044]
pitches = [1.0, 2.0]
lengths = [10., 20.]

smFrameSize = 400
slitColumnSpacing = 3.

# Create the smaller write field and corresponding markers
smField1 = Frame("SmallField1", (smFrameSize, smFrameSize), [])
smField1.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField1.make_slit_array(pitches[0], slitColumnSpacing, widths, lengths[0], rotAngle, 100, 100, 30, l_smBeam)

smField2 = Frame("SmallField2", (smFrameSize, smFrameSize), [])
smField2.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField2.make_slit_array(pitches[0], slitColumnSpacing, widths, lengths[1], rotAngle, 100, 100, 30, l_smBeam)

smField3 = Frame("SmallField3", (smFrameSize, smFrameSize), [])
smField3.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField3.make_slit_array(pitches[1], slitColumnSpacing, widths, lengths[0], rotAngle, 100, 100, 30, l_smBeam)

smField4 = Frame("SmallField4", (smFrameSize, smFrameSize), [])
smField4.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)
smField4.make_slit_array(pitches[1], slitColumnSpacing, widths, lengths[1], rotAngle, 100, 100, 30, l_smBeam)

centerAlignField = Frame("CenterAlignField", (smFrameSize, smFrameSize), [])
centerAlignField.make_align_markers(2., 20., (180., 180.), l_lgBeam, cross=True)

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
if putOnWafer:  # Fit as many patterns on a wafer as possible
    wafer = MBEWafer('MembranesWafer', wafer_r=wafer_r, cells=[topCell], cell_gap=CELL_GAP, mkWidth=tDicingMarks,
                     cellsAtEdges=False, symmetric_chips=True)
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

# layout.show()
else:  # Only output a single copy of the pattern (not on a wafer)
    layout.add(topCell)
    layout.show()

file_string = str(waferVer) + '_' + str(density) + ' dMark' + str(tDicingMarks)
filename = file_string.replace(' ', '_') + '.gds'
layout.save(filename)
