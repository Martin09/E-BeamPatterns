# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:04:48 2016

@author: Martin Friedl
"""
import numpy as np

from gdsCAD_py3.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_py3.shapes import Box, Rectangle, Label

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

    def makeYShape(self, length, width, rotAngle, spacing, Nx, Ny, layers):
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
            self.add(allshapes)

    def makeTriShape(self, length, width, rotAngle, spacing, Nx, Ny, layers):
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
            self.add(allshapes)

    def makeXShape(self, length, width, rotAngle, spacing, Nx, Ny, layers):
        if not (type(layers) == list): layers = [layers]
        pt1 = np.array((-length / 2., -width / 2.))
        pt2 = np.array((length / 2., width / 2.))
        slit = Cell("Slit")
        for l in layers:
            rect = Rectangle(pt1, pt2, layer=l)
            slit.add(rect)
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


def makeSlitArray(pitches, spacing, widths, lengths, rotAngle,
                  arrayHeight, arraySpacing, layers):
    '''
    Give it a single pitch and width and it will generate an array for all the lengths
    '''
    if not (type(layers) == list): layers = [layers]
    if not (type(pitches) == list): pitches = [pitches]
    if not (type(lengths) == list): lengths = [lengths]
    if not (type(widths) == list): widths = [widths]
    for l in layers:
        i = -1
        j = -1
        manyslits = Cell("SlitArray")
        slitarray = Cell("SlitArray")
        pitch = pitches[0]
        width = widths[0]
        j += 1
        i = -1
        xlength = 0
        slit = Cell("Slits")
        for length in lengths:
            spacing = length / 5. + 0.1
            i += 1
            pitchV = pitch / np.cos(np.deg2rad(rotAngle))
            #            widthV = width / np.cos(np.deg2rad(rotAngle))
            #            Nx = int(arrayWidth / (length + spacing))
            Ny = int(arrayHeight / (pitchV))
            # Define the slits
            if xlength == 0:
                translation = (length / 2., 0)
                xlength += length
            else:
                translation = (xlength + spacing + length / 2., 0)
                xlength += length + spacing

            pt1 = np.array((-length / 2., -width / 2.)) + translation
            pt2 = np.array((length / 2., width / 2.)) + translation
            rect = Rectangle(pt1, pt2, layer=l)
            rect = rect.copy().rotate(rotAngle)
            slit.add(rect)
        slits = CellArray(slit, 1, Ny, (0, pitchV))
        # slits.translate((-(Nx - 1) * (length + spacing) / 2., -(Ny - 1)* (pitchV) / 2.))
        slits.translate((-slits.bounding_box[1, 0] / 2., -slits.bounding_box[1, 1] / 2.))

        slitarray.add(slits)
        text = Label('w/p\n%i/%i' % (width * 1000, pitch * 1000), 2, layer=l_smBeam)
        lblVertOffset = 1.4
        text.translate(tuple(
            np.array(-text.bounding_box.mean(0)) + np.array((0, arrayHeight / lblVertOffset))))  # Center justify label
        slitarray.add(text)
        #            manyslits.add(slitarray,origin=((arrayWidth + arraySpacing) * i, (arrayHeight + 2.*arraySpacing) * j-arraySpacing/2.))
        manyslits.add(slitarray)

    # self.add(manyslits, origin=(-i * (arrayWidth + arraySpacing) / 2, -j * (arrayHeight + arraySpacing) / 2))
    #    self.add(manyslits)
    return manyslits


def makeSlitArray2(pitches, spacing, widths, lengths, rotAngle,
                   arrayHeight, arrayWidth, arraySpacing, layers):
    '''
    Give it a single pitch and lengths/widths and it will generate an array for all the combinations
    Makes seperate frame for each length value
    '''
    if not (type(layers) == list): layers = [layers]
    if not (type(pitches) == list): pitches = [pitches]
    if not (type(lengths) == list): lengths = [lengths]
    if not (type(widths) == list): widths = [widths]
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

                pitchV = pitch
                #                    widthV = width / np.cos(np.deg2rad(rotAngle))
                Nx = int(arrayWidth / (length + spacing))
                Ny = int(arrayHeight / (pitchV))
                # Define the slits
                slit = Cell("Slits")
                line = Path([[-length / 2., 0], [length / 2., 0]], width=width, layer=l)
                slit.add(line)
                slits = CellArray(slit, Nx, Ny, (length + spacing, pitchV))
                slits.translate((-(Nx - 1) * (length + spacing) / 2., -(Ny - 1) * (pitchV) / 2.))
                slitarray = Cell("SlitArray")
                slitarray.add(slits)
                text = Label('w/p/l\n%i/%i/%i' % (width * 1000, pitch * 1000, length), 1, layer=l_smBeam)
                lblVertOffset = 1.5
                if j % 2 == 0:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            arrayWidth / 2., -arrayHeight / lblVertOffset))))  # Center justify label
                else:
                    text.translate(
                        tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                            arrayWidth / 2., arrayHeight / lblVertOffset))))  # Center justify label
                slitarray.add(text)
                manyslits.add(slitarray,
                              origin=((arrayWidth + arraySpacing) * i, (
                                      arrayHeight + arraySpacing) * j - arraySpacing))
    return manyslits


def make_theory_cell(wafer_orient='111'):
    ''' Makes the theory cell and returns ir as a cell'''

    # Pitch Dependence
    PitchDep = Cell('PitchDependence')
    arrayHeight = 20.
    arrayWidth = 20.
    arraySpacing = 10.
    spacing = 0.5

    length = [arrayWidth]
    widths = [0.020, 0.040, 0.080, 0.140, 0.220, 0.320]
    #widths = [0.008, 0.016, 0.024, 0.032, 0.040, 0.048]
    pitches = [1.0, 2.0, 4.0]

    for j, width in enumerate(widths):
        for i, pitch in enumerate(pitches):
            PitchDep.add(
                makeSlitArray2(pitch, spacing, width, length, 0, arrayHeight, arrayWidth, arraySpacing, l_smBeam),
                origin=(i * 1.5 * arrayWidth, j * 1.5 * arrayHeight))

    TopCell = Cell('GrowthTheoryTopCell')
    TopCell.add(PitchDep, origin=(0., 0.))
    # # TODO: Add the branched growth shapes

    return TopCell


if __name__ == "__main__":
    TopCell = make_theory_cell()
    # Add the copied cell to a Layout and save
    layout = Layout('LIBRARY')
    layout.add(TopCell)
    layout.save('NMDoseTest_v2.gds')
