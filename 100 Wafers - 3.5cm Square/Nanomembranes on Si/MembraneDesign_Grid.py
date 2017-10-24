# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:11:31 2015

@author: Martin Friedl
"""

from datetime import timedelta, date

import numpy as np
from GrowthTheoryCell_v2 import make_theory_cell
from gdsCAD_v045.core import Cell, Boundary, CellArray, Layout
from gdsCAD_v045.shapes import Box, Rectangle, Label
from shapely.affinity import rotate as rotateshape
from shapely.geometry import LineString
from gdsCAD_v045.templates_onSi import Wafer_GridStyle

putOnWafer = True  # Output full wafer or just a single pattern?
HighDensity = False  # High density of triangles?
glbAlignmentMarks = False
tDicingMarks = 10.  # Dicing mark line thickness (um)
rotAngle = 0.  # Rotation angle of the membranes
wafer_r = 50e3
waferVer = 'Membranes on Si v1.1 r{:d}'.format(int(wafer_r / 1000))
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


class Frame(Cell):
    """
    Make a frame for writing to with ebeam lithography
    Params:
    -name of the frame, just like when naming a cell
    -size: the size of the frame as an array [xsize,ysize]
    """

    def __init__(self, name, size, borderLayers):
        if not (type(borderLayers) == list):
            borderLayers = [borderLayers]
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
        if not (type(layers) == list):
            layers = [layers]
        self.aMarkers = Cell("AlignMarkers")
        for l in layers:
            if not (cross):
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
            self.aMarkers.add([am1, am2, am3, am4])
            self.add(self.aMarkers)

    def makeSlitArray(self, pitches, spacing, widths, lengths, rotAngle,
                      arrayHeight, arrayWidth, arraySpacing, layers):
        if not (type(layers) == list):
            layers = [layers]
        if not (type(pitches) == list):
            pitches = [pitches]
        if not (type(lengths) == list):
            lengths = [lengths]
        if not (type(widths) == list):
            widths = [widths]
        for l in layers:
            i = -1
            j = -1
            manyslits = Cell("SlitArray")
            pitch = pitches[0]
            for length in lengths:
                j += 1
                i = -1

                for width in widths:
                    #            for pitch in pitches:
                    i += 1
                    if i % 3 == 0:
                        j += 1  # Move to array to next line
                        i = 0  # Restart at left

                    pitchV = pitch / np.cos(np.deg2rad(rotAngle))
                    #                    widthV = width / np.cos(np.deg2rad(rotAngle))
                    Nx = int(arrayWidth / (length + spacing))
                    Ny = int(arrayHeight / (pitchV))
                    # Define the slits
                    slit = Cell("Slits")
                    rect = Rectangle(
                        (-length / 2., -width / 2.),
                        (length / 2., width / 2.),
                        layer=l)
                    rect = rect.copy().rotate(rotAngle)
                    slit.add(rect)
                    slits = CellArray(slit, Nx, Ny,
                                      (length + spacing, pitchV))
                    slits.translate((-(Nx - 1) * (length + spacing) / 2., -(Ny - 1) * (pitchV) / 2.))
                    slitarray = Cell("SlitArray")
                    slitarray.add(slits)
                    text = Label('w/p/l\n%i/%i/%i' %
                                 (width * 1000, pitch, length), 5)
                    lblVertOffset = 1.35
                    if j % 2 == 0:
                        text.translate(
                            tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                                0, -arrayHeight / lblVertOffset))))  # Center justify label
                    else:
                        text.translate(
                            tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                                0, arrayHeight / lblVertOffset))))  # Center justify label
                    slitarray.add(text)
                    manyslits.add(slitarray,
                                  origin=((arrayWidth + arraySpacing) * i, (
                                      arrayHeight + 2. * arraySpacing) * j - arraySpacing / 2.))

        self.add(manyslits,
                 origin=(-i * (arrayWidth + arraySpacing) / 2, -(j + 1.5) * (
                     arrayHeight + arraySpacing) / 2))


# %%Create the pattern that we want to write

lgField = Frame("LargeField", (2000., 2000.), [])  # Create the large write field
lgField.makeAlignMarkers(20., 200., (850., 850.), l_lgBeam, cross=True)

# Define parameters that we will use for the slits
widths = [0.004, 0.008, 0.012, 0.016, 0.028, 0.044]
pitches = [1.0, 2.0]
lengths = [10., 20.]

smFrameSize = 400
slitColumnSpacing = 3.

# Create the smaller write field and corresponding markers
smField1 = Frame("SmallField1", (smFrameSize, smFrameSize), [])
smField1.makeAlignMarkers(2., 20., (180., 180.), l_lgBeam, cross=False)
smField1.makeSlitArray(pitches[0], slitColumnSpacing, widths, lengths[0], rotAngle, 100, 100, 30, l_smBeam)

smField2 = Frame("SmallField2", (smFrameSize, smFrameSize), [])
smField2.makeAlignMarkers(2., 20., (180., 180.), l_lgBeam, cross=False)
smField2.makeSlitArray(pitches[0], slitColumnSpacing, widths, lengths[1], rotAngle, 100, 100, 30, l_smBeam)

smField3 = Frame("SmallField3", (smFrameSize, smFrameSize), [])
smField3.makeAlignMarkers(2., 20., (180., 180.), l_lgBeam, cross=False)
smField3.makeSlitArray(pitches[1], slitColumnSpacing, widths, lengths[0], rotAngle, 100, 100, 30, l_smBeam)

smField4 = Frame("SmallField4", (smFrameSize, smFrameSize), [])
smField4.makeAlignMarkers(2., 20., (180., 180.), l_lgBeam, cross=False)
smField4.makeSlitArray(pitches[1], slitColumnSpacing, widths, lengths[1], rotAngle, 100, 100, 30, l_smBeam)

centerAlignField = Frame("CenterAlignField", (smFrameSize, smFrameSize), [])
centerAlignField.makeAlignMarkers(2., 20., (180., 180.), l_lgBeam, cross=False)

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

theory_cell = make_theory_cell()
topCell.add(theory_cell, origin=(65., 55.))

# topCellArray = CellArray(topCell,1,1,(0,0),rotation=90)
# topCell = Cell('RotatedTopCell')
# topCell.add(topCellArray)

# %%Create the layout and output GDS file
layout = Layout('LIBRARY')
if putOnWafer:  # Fit as many patterns on a wafer as possible
    # wafer = MBEWafer('MembranesWafer',wafer_r=wafer_r,cells=[topCell], cell_gap=CELL_GAP, mkWidth=tDicingMarks,cellsAtEdges=False)
    wafer = Wafer_GridStyle('MembranesWafer_GridStyle', cells=[topCell], block_gap=18000)  # 28000)
    wafer.wafer_r = wafer_r
    wafer.block_size = np.array([34.7e3, 34.7e3])
    wafer._place_blocks()
    wafer.add_wafer_outline()
    # wafer.add_dicing_marks()
    wafer.add_dicing_crosses()
    lblx = 17350 - 867
    lbly = 17350
    wafer.o_text = {'A': (lblx, lbly - 1857), 'B': (-lblx, lbly - 1857), 'C': (-lblx, -lbly - 1857),
                    'D': (lblx, -lbly - 1857)}
    wafer.add_orientation_text()
    wafer.add_blocks()
    # TODO: Remove the alignment crosses in the layout
    # TODO: try to remove the cropped fields and make it simply a 2x2 layout
    layout.add(wafer)
# layout.show()
else:  # Only output a single copy of the pattern (not on a wafer)
    layout.add(topCell)
    layout.show()

filestring = str(waferVer) + '_' + str(density) + ' dMark' + str(tDicingMarks)
filename = filestring.replace(' ', '_') + '.gds'
layout.save(filename)

# freq = 20E6 #20 GHz
# spotsize = 100E-9 #100nm beam
# gridsize = np.sqrt(2)/2.*spotsize
# spotarea = gridsize**2.
# waferarea = wafer.area()/1E6**2.
# writetime = waferarea/spotarea/freq
# time = timedelta(seconds=writetime)
# print '\nEstimated write time: \n'+str(time)
