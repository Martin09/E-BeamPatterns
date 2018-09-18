# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 10:04:48 2016

@author: Martin Friedl
"""
import numpy as np
from gdsCAD_py3.shapes import Box, Rectangle, Label, Disk, RegPolygon
from gdsCAD_py3.core import Cell, Boundary, CellArray, Layout, Path
from gdsCAD_py3.templates111 import Wafer_TriangStyle, dashed_line
from datetime import timedelta, date

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
                text = Label('w/p/l\n%i/%i/%i' % (width * 1000, pitch * 1000, length * 1000), 2, layer=l_smBeam)
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
    return manyslits


#    self.add(manyslits,
#             origin=(-i * (arrayWidth + arraySpacing) / 2, -(j+1.5) * (
#                 arrayHeight + arraySpacing) / 2))


def makeSlitArray3(pitches, spacing, widths, lengths, rotAngle,
                   arrayHeight, arrayWidth, arraySpacing, layers):
    '''
    Give it a single pitch and arrays for spacings/widths and it will generate an array for all the combinations
    Makes seperate frame for each pitch
    '''
    if not (type(layers) == list): layers = [layers]
    if not (type(pitches) == list): pitches = [pitches]
    if not (type(lengths) == list): lengths = [lengths]
    if not (type(widths) == list): widths = [widths]
    for l in layers:
        i = -1
        j = -1
        manyslits = Cell("SlitArray")
        length = lengths[0]
        spacing = length / 5. + 0.1  # Set the spacing between arrays
        for pitch in pitches:
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
                text = Label('w/p/l\n%i/%i/%i' % (width * 1000, pitch * 1000, length * 1000), 2, layer=l_smBeam)
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
    return manyslits


#    self.add(manyslits,
#             origin=(-i * (arrayWidth + arraySpacing) / 2, -(j+1.5) * (
#                 arrayHeight + arraySpacing) / 2))


def make_rotating_slits(length, width, N, radius, layers, angleRef=None):
    """

    :param length: Length of the slits in the circle
    :param width: Width of the slits in the circle
    :param N: Number of slits going around the circle
    :param radius: Radius of the circle
    :param layers: Layers to write the slits in
    :param angleRef: if None, no angle reference lines are added. If '111' then add reference lines at 30/60 degrees. If '100' then add reference lines at 45/90 degrees.
    :return:
    """
    cell = Cell('RotatingSlits')
    if not (type(layers) == list): layers = [layers]
    allslits = Cell('All Slits')
    angles = np.linspace(0, 360, N)
    #        radius = length*12.
    translation = (radius, 0)
    pt1 = np.array((-length / 2., -width / 2.)) + translation
    pt2 = np.array((length / 2., width / 2.)) + translation
    slit = Cell("Slit")
    for l in layers:
        rect = Rectangle(pt1, pt2, layer=l)
        slit.add(rect)
        for angle in angles:
            allslits.add(slit.copy(), rotation=angle)
        cell.add(allslits)

        if angleRef:
            labelCell = Cell('AngleLabels')
            lineCell = Cell('Line')
            pt1 = (-radius * 0.9, 0)
            pt2 = (radius * 0.9, 0)
            line = Path([pt1, pt2], width=width, layer=l)
            dLine = dashed_line(pt1, pt2, 2, width, l)
            lineCell.add(line)
            labelCell.add(lineCell, rotation=0)
            if angleRef == '111':
                labelCell.add(lineCell, rotation=60)
                labelCell.add(lineCell, rotation=-60)
                labelCell.add(dLine, rotation=30)
                labelCell.add(dLine, rotation=90)
                labelCell.add(dLine, rotation=-30)
            elif angleRef == '100':
                labelCell.add(lineCell, rotation=0)
                labelCell.add(lineCell, rotation=90)
                labelCell.add(dLine, rotation=45)
                labelCell.add(dLine, rotation=135)
            cell.add(labelCell)

        return cell


# TODO: Center array around the origin
def make_shape_array(array_size, shape_area, shape_pitch, type, layer, labels=True):
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

    if labels:
        text = Label('{}'.format(type), 2, layer=l_smBeam)
        lblVertOffset = 0.8
        text.translate(
            tuple(np.array(-text.bounding_box.mean(0)) + np.array((
                array_size / 2., array_size / lblVertOffset))))  # Center justify label

        shape_array_cell.add(text)

    return shape_array_cell


def make_many_shapes(array_size, shape_areas, pitch, shapes, layer):
    offset_x = array_size * 1.25
    offset_y = array_size * 1.25
    cur_y = 0
    many_shape_cell = Cell('ManyShapes')
    for area in shape_areas:
        cur_x = 0
        for shape in shapes:
            write_labels = cur_y == 0
            s_array = make_shape_array(array_size, area, pitch, shape, layer, labels=write_labels)
            many_shape_cell.add(s_array, origin=(cur_x, cur_y))
            cur_x += offset_x
        cur_y -= offset_y
    return many_shape_cell


def make_theory_cell(wafer_orient = '111'):
    ''' Makes the theory cell and returns ir as a cell'''
    # Growth Theory Slit Elongation
    pitch = [0.500]
    lengths = list(np.logspace(-3, 0, 20) * 8.0)  # Logarithmic
    widths = [0.044, 0.028, 0.016, 0.012, 0.008]
    TheorySlitElong = Cell('LenWidthDependence')
    arrayHeight = 20.
    arraySpacing = 30.
    spacing = 10.

    TheorySlitElong.add(makeSlitArray(pitch, spacing, widths[0], lengths, 0., arrayHeight, arraySpacing, l_smBeam))
    TheorySlitElong.add(makeSlitArray(pitch, spacing, widths[1], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
                        origin=(0, -30))
    TheorySlitElong.add(makeSlitArray(pitch, spacing, widths[2], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
                        origin=(0, -60))
    TheorySlitElong.add(makeSlitArray(pitch, spacing, widths[3], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
                        origin=(0, -90))
    TheorySlitElong.add(makeSlitArray(pitch, spacing, widths[4], lengths, 0., arrayHeight, arraySpacing, l_smBeam),
                        origin=(0, -120))

    # Length Dependence
    LenWidDep = Cell('LenWidDependence')
    pitch = [1.0]
    # lengths = list(np.logspace(-3, 0, 10) * 8.0)  # Logarithmic
    lengths = [0.008, 0.037, 0.170, 0.370, 0.800, 1.300, 1.700, 2.700, 3.700, 8.000]
    widths = [0.044, 0.016, 0.008]
    arrayHeight = 20.
    arrayWidth = arrayHeight
    arraySpacing = 30.
    spacing = 0.5

    for i, length in enumerate(lengths):
        for j, width in enumerate(widths):
            LenWidDep.add(
                makeSlitArray3(pitch, spacing, width, length, 0, arrayHeight, arrayWidth, arraySpacing, l_smBeam),
                origin=(i * 30, j * 30))

    # Second length dependence with pitch 500nm
    pitch = [0.5]
    LenWidDep2 = Cell('LenWidDependence2')
    for i, length in enumerate(lengths):
        for j, width in enumerate(widths):
            LenWidDep2.add(
                makeSlitArray3(pitch, spacing, width, length, 0, arrayHeight, arrayWidth, arraySpacing, l_smBeam),
                origin=(i * 30, j * 30))


    # Make rotating slits
    wheel1 = Cell('RotDependence_LongSlits')
    wheel1.add(make_rotating_slits(5, 0.044, 361, 6. * 5, l_smBeam, angleRef=True))
    wheel1.add(make_rotating_slits(5, 0.044, 433, 7.2 * 5, l_smBeam))
    wheel1.add(make_rotating_slits(5, 0.044, 505, 8.4 * 5, l_smBeam))

    wheel2 = Cell('RotDependence_ShortSlits')
    wheel2.add(make_rotating_slits(2, 0.044, 200, 6. * 2, l_smBeam, angleRef=True))
    for i in range(10):  # number of concentric rings to make
        wheel2.add(make_rotating_slits(2, 0.044, 200, (7.2 + i * 1.2) * 2, l_smBeam))

    # Pitch Dependence
    PitchDep = Cell('PitchDependence')
    pitches = list(np.round(np.logspace(-1, 1, 10), 1))  # Logarithmic
    length = [3.]
    widths = [0.054, 0.044, 0.028, 0.016, 0.008]
    arrayHeight = 20.
    arrayWidth = arrayHeight
    arraySpacing = 30.
    spacing = 0.5

    for j, width in enumerate(widths):
        for i, pitch in enumerate(pitches):
            PitchDep.add(
                makeSlitArray2(pitch, spacing, width, length, 0, arrayHeight, arrayWidth, arraySpacing, l_smBeam),
                origin=(i * 30, j * 30))
    # Make arrays of various shapes
    hexagon_array = make_shape_array(20, 0.02, 0.75, 'Hexagons', l_smBeam)
    circles_array = make_shape_array(20, 0.02, 0.75, 'Circles', l_smBeam)
    triangle_down_array = make_shape_array(20, 0.02, 0.75, 'Tris_down', l_smBeam)
    triangle_up_array = make_shape_array(20, 0.02, 0.75, 'Tris_up', l_smBeam)

    TopCell = Cell('GrowthTheoryTopCell')
    TopCell.add(wheel1, origin=(-100., -100.))
    TopCell.add(wheel2, origin=(0., -100.))
    TopCell.add(PitchDep, origin=(-200., -350.))
    TopCell.add(TheorySlitElong, origin=(-250., -50))
    TopCell.add(LenWidDep, origin=(-200., -50.))
    TopCell.add(LenWidDep2, origin=(-200., 50.))
    TopCell.add(hexagon_array, origin=(-100., -40))
    TopCell.add(circles_array, origin=(-75., -40.))
    TopCell.add(triangle_down_array, origin=(-50., -40))
    TopCell.add(triangle_up_array, origin=(-25., -40))

    # TODO: Add the branched growth shapes

    return TopCell


if __name__ == "__main__":
    TopCell = make_theory_cell()
    # Add the copied cell to a Layout and save
    layout = Layout('LIBRARY')
    layout.add(TopCell)
    layout.save('GrowthTheoryCell.gds')
