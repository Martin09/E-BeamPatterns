# -*- coding: utf-8 -*-
"""
Templates for automating the design of different wafer styles.

.. note::
    Copyright 2009-2012 Lucas Heitzmann Gabrielli
    
    Copyright 2013 Andrew G. Mark

    gdsCAD (based on gdspy) is released under the terms of the GNU GPL
    
"""
# TODO: Make it more pythonic, create separate classes for cells, blocks etc. to make it easier to read
import string
from operator import itemgetter

import networkx as nx
import numpy as np
from descartes.patch import PolygonPatch
from gdsCAD.core import Cell, Path, Boundary
from gdsCAD.shapes import Circle, Label, LineLabel
from shapely.affinity import rotate as rotateshape
from shapely.affinity import translate as translateshape
from shapely.geometry import Polygon, Point, LineString, box


# Helper function:
# Given two points from a line, returns a cell containing a dashed line connecting the two points
def dashed_line(pt1, pt2, dashlength, width, layer):
    line = LineString((pt1, pt2))
    dash_pts = np.arange(0, line.length, dashlength).tolist()
    if len(dash_pts) % 2 == 1:  # Odd number
        dash_pts.append(line.length)  # Add last point on line
    dash_pts = map(line.interpolate, dash_pts)  # Interpolate points along this line to make dashes
    dash_pts = [pt.xy for pt in dash_pts]
    dash_pts = np.reshape(dash_pts, (-1, 2, 2))
    lines = [Path(list(linepts), width=width, layer=layer) for linepts in dash_pts]
    dline = Cell('DASHLINE')
    dline.add(lines)
    return dline


class Wafer_TriangStyle(Cell):
    """
    Wafer style for [111] wafers consisting of triangular blocks of patterned features.
    
    :param name: The name of the new wafer cell
    :param wafer_r: the radius of the wafer in um
    :param cells:  a list of cells that will be tiled to fill each blocks
                   the cells will be cycled until all blocks are filled.
    :param block_gap: the gap between the triangular blocks
    :param cell_gap: the gap between the square cells within each block
    :param trisize: in um, length of triangle sides
    :param cellsize: size of each cell within a block
    :param MCIterations: Number of monte carlo iterations used to find optimal position of cells within the blocks
    :param doMCSearch: Whether or not to optimize the placement of the square cells within each triangular block
    :param block_gap: the distance to leave between blocks
    :param symmetric_chips: makes the up-facing and down-facing chips symmetric by rotating them 180 degrees. However for direction-sensitive devices (ex: branched structures) the 180 degree rotation is undersirable.

    :returns: A new wafer ``Cell``        

    Spacing between cells in a block is determined automatically based on the cell
    bounding box, or by using the attribute cell.spacing if it is available.

    """

    # the placement of the wafer alignment points
    align_pts = None

    def __init__(self,
                 name,
                 cells=None,
                 wafer_r=25.5e3,
                 trisize=10e3,
                 cellsize=2e3,
                 block_gap=0.,
                 cell_gap=200.,
                 doMCSearch=True,
                 MCIterations=30,  # Small square cells
                 doMCBlockSearch=True,
                 MCBlockIterations=50,  # Large triangular blocks
                 mkWidth=10,
                 cellsAtEdges=False,
                 symmetric_chips=True):
        Cell.__init__(self, name)
        self.wafer_r = wafer_r
        self.trisize = trisize
        self.cellsize = cellsize
        self.block_gap = block_gap
        self.cell_gap = cell_gap
        self.doMCSearch = doMCSearch
        self.MCIterations = MCIterations
        self.doMCBlockSearch = doMCBlockSearch
        self.MCBlockIterations = MCBlockIterations
        # Create a circle shape with the radius of the wafer
        circ = Point(0., 0.)
        self.waferShape = circ.buffer(wafer_r)
        self.blockOffset = (0, 0)
        self.cells = cells
        self.cell_layers = self._cell_layers()
        self._label = None
        self.upCellLattice = []
        self.downCellLattice = []
        self.upCenters = []
        self.downCenters = []
        self.upTris = []
        self.downTris = []
        self.cellsAtEdges = cellsAtEdges
        self.symmetric_chips = symmetric_chips

    def _cell_layers(self):
        """
        A list of all active layers in ``cells``
        """
        cell_layers = set()
        for c in self.cells:
            if isinstance(c, Cell):
                cell_layers |= set(c.get_layers())
            else:
                for s in c:
                    cell_layers |= set(s.get_layers())
        return list(cell_layers)

    def add_aligment_marks(self, layers):
        """
        Create alignment marks on all active layers
        """
        if not (type(layers) == list): layers = [layers]
        d_layers = self.cell_layers
        #        styles=['B' if i%2 else 'B' for i in range(len(d_layers))]            
        #        am = AlignmentMarks(styles, d_layers)
        am = Cell('CONT_ALGN')
        # Define dimensions of the alignment cross
        t = 200.  # Thickness
        t /= 2.
        h = 2000.  # Height
        w = 2000.  # Width
        crosspts = [
            (-t, t), (-t, h), (t, h), (t, t), (w, t), (w, -t), (t, -t), (t, -h),
            (-t, -h), (-t, -t), (-w, -t), (-w, t)]
        # Create shapely polygon for later calculation
        crossPolygon = Polygon(crosspts)
        crossPolygons = []
        for pt in self.align_pts:
            crossPolygons.extend([
                translateshape(crossPolygon, xoff=pt[0], yoff=pt[1])])

        # TODO: Replace these two loops with a single loop, looping over an array of block objects
        # TODO: Make the deleting more efficient by using del for multiple indexes?
        i_del = []
        # Loop over all triangular blocks
        for i, tri in enumerate(self.upTris):
            for poly in crossPolygons:  # Loop over all alignment crosses
                if poly.intersects(tri) or poly.within(tri) or poly.contains(
                        tri):
                    # If conflict is detected, remove that triangular block
                    i_del.append(i)
                    print 'up:' + str(self.upTris[i].centroid.xy)

        self.upTris = [tri for i, tri in enumerate(self.upTris) if i not in i_del]

        # Repeat for down-facing triangles
        i_del = []
        for i, tri in enumerate(self.downTris):
            for poly in crossPolygons:
                if poly.intersects(tri) or poly.within(tri) or poly.contains(
                        tri):
                    # If conflict is detected, remove that triangular block
                    i_del.append(i)
                    print 'down:' + str(self.downTris[i].centroid.xy)

        self.downTris = [tri for i, tri in enumerate(self.downTris) if i not in i_del]
        # Refresh the centers of the remaining triangles
        self.upCenters = [list(zip(*tri.centroid.xy)[0]) for tri in self.upTris]
        self.downCenters = [list(zip(*tri.centroid.xy)[0])
                            for tri in self.downTris]

        for l in layers:  # Add marker to all layers
            cross = Boundary(crosspts, layer=l)  # Create gdsCAD shape
            am.add(cross)

        mblock = Cell('WAF_ALGN_BLKS')
        mblock.add(am)
        for pt in self.align_pts:
            self.add(mblock, origin=pt)

    def add_orientation_text(self, layers):
        """
        Create Orientation Label
        """
        if not (type(layers) == list): layers = [layers]
        tblock = Cell('WAF_ORI_TEXT')
        for l in layers:
            for (t, pt) in self.o_text.iteritems():
                txt = Label(t, 1000, layer=l)
                bbox = txt.bounding_box
                txt.translate(-np.mean(bbox, 0))  # Center text around origin
                txt.translate(np.array(pt))
                tblock.add(txt)
        self.add(tblock)

    def add_dicing_marks(self, layers, mkWidth=100):
        """
        Create dicing marks
        """
        if not (type(layers) == list): layers = [layers]
        # Define the array and wafer parameters
        gap = self.block_gap
        wafer_r = self.wafer_r
        sl_tri = self.trisize
        sl_lattice = sl_tri + gap / np.tan(np.deg2rad(30))
        h_lattice = np.sqrt(3.) / 2. * sl_lattice
        # Create the lattice of the "up" facing triangles
        points = self.createPtLattice(2. * wafer_r, sl_lattice / 2., h_lattice)
        points = [np.array(elem) for elem in points]
        points = points + np.array(
            [-sl_lattice / 2., 0
             ])  # Shift lattice so we can cleave the wafer at y=0
        points = points + np.array(self.blockOffset)  # Shift by point from MC search (if applicable)
        points = [
            point
            for point in points
            if (Point(point).distance(Point(0, 0)) < wafer_r)
            ]
        import pylab as plt
        # Plot points
        x, y = zip(*points)
        plt.plot(x, y, 'ko')
        # Create a lineshape of the boundary of the circle
        c = self.waferShape.boundary
        # Create a set (unordered with unique entries)
        dicinglines = set()
        # For each point in the lattice, create three lines (one along each direction)
        for x, y in points:
            l0 = LineString([(-4. * wafer_r, y), (4. * wafer_r, y)])
            l1 = rotateshape(l0, 60, origin=(x, y))
            l2 = rotateshape(l0, -60, origin=(x, y))
            # See where these lines intersect the wafer outline
            i0 = c.intersection(l0)
            i1 = c.intersection(l1)
            i2 = c.intersection(l2)
            p0s = tuple(map(tuple, np.round((i0.geoms[0].coords[0], i0.geoms[
                1].coords[0]))))
            p1s = tuple(map(tuple, np.round((i1.geoms[0].coords[0], i1.geoms[
                1].coords[0]))))
            p2s = tuple(map(tuple, np.round((i2.geoms[0].coords[0], i2.geoms[
                1].coords[0]))))
            # Add these points to a unique unordered set
            dicinglines.add(p0s)
            dicinglines.add(p1s)
            dicinglines.add(p2s)
        # At the end of the loop, the set will contain a list of point pairs which can be uesd to make the dicing marks
        dmarks = Cell('DIC_MRKS')
        for l in layers:
            for p1, p2 in dicinglines:
                dicingline = Path([p1, p2], width=mkWidth, layer=l)
                dmarks.add(dicingline)
            self.add(dmarks)
        return points

    def add_wafer_outline(self, layers):
        """
        Create Wafer Outline
        """
        if not (type(layers) == list): layers = [layers]

        outline = Cell('WAF_OLINE')
        for l in layers:
            circ = Circle((0, 0), self.wafer_r, 100, layer=l)
            outline.add(circ)
        self.add(outline)

    # Gets an optimized list of points where the cells will then be projected within each block
    def getCellLattice(self, cellsize=2000):
        iterations = self.MCIterations
        ycelloffset = self.cell_gap / 3.5  # Arbitrary, change by trial and error
        if self.doMCSearch:
            best = [0, 0, 0, 0]
            # Iterates many times to find the best fit
            for i in range(iterations):
                # Random seed point
                rndpt = (0, np.random.randint(-cellsize, cellsize))
                # Make cells around this point
                cells = self.makeCells(startpt=rndpt, cellsize=cellsize)
                if not cells:
                    continue
                centroidDist = np.array([cell.centroid.xy for cell in cells]).squeeze()
                if len(centroidDist.shape) == 2:
                    centroidDist = centroidDist.mean(0)
                if len(cells) > best[1]:
                    best = [rndpt, len(cells), cells, centroidDist]
                elif len(cells) == best[1]:
                    # Choose the one that is closer to the center of the wafer
                    if np.sqrt(rndpt[0] ** 2 + rndpt[1] ** 2) < np.sqrt(best[0][0] ** 2 + best[0][1] ** 2):
                        #                    if centroidDist < best[3]:
                        best = [rndpt, len(cells), cells, centroidDist]
                        #                print("Current: {:f}, Best {:f}").format(len(cells),best[1])

                        #            centroidDist = np.array([tri.centroid.xy for tri in cells]).squeeze().mean(0)
                        #            centroidDist = np.sqrt(centroidDist[0]**2+centroidDist[1]**2)
                        #            centroidDist = np.array([cell.centroid.xy for cell in cells]).squeeze()

            # Choose the best configuration (fits the most cells and is closest to centroid)
            cells = best[2]
        else:
            cells = self.makeCells(cellsize=2000)
        sl_tri = self.trisize
        h_tri = np.sqrt(3.) / 2. * sl_tri
        gap = self.block_gap
        from matplotlib import pyplot
        fig = pyplot.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.grid()
        ax.axis('equal')
        block = Polygon([
            [-sl_tri / 2., -h_tri / 3.], [sl_tri / 2., -h_tri / 3.],
            [0, 2. * h_tri / 3.], [-sl_tri / 2., -h_tri / 3.]
        ])
        block = translateshape(block, yoff=h_tri / 3. + gap / 2.)
        block = translateshape(block, xoff=self.blockOffset[0],
                               yoff=self.blockOffset[1])  # TODO: plot output not working properly because of this?
        patch = PolygonPatch(block,
                             facecolor="#{0:0{1}X}".format(np.random.randint(0, 16777215), 6),
                             #                             facecolor='r',
                             edgecolor='k',
                             alpha=0.3,
                             zorder=2)
        ax.add_patch(patch)
        ax.plot(block.exterior.coords.xy[0], block.exterior.coords.xy[1], 'k-')
        for cell in cells:
            cell = translateshape(cell, yoff=h_tri / 3. + gap / 2. + ycelloffset)
            cell = translateshape(cell, xoff=self.blockOffset[0],
                                  yoff=self.blockOffset[1])  # TODO: plot output not working properly because of this?
            patch = PolygonPatch(cell,
                                 facecolor="#{0:0{1}X}".format(np.random.randint(0, 16777215), 6),
                                 edgecolor='k',
                                 alpha=0.3,
                                 zorder=2)
            ax.add_patch(patch)
        # Convert cells to lattice of points
        cellLattice = np.array([zip(*cell.centroid.xy)[0] for cell in cells])
        cellLattice = cellLattice + np.array([0, ycelloffset])
        return cellLattice

    # Make takes square cells and sees how many can be fit into a triangular block
    def makeCells(self, cellsize=2000, startpt=(0, 0)):
        gap = self.cell_gap
        # Define the parameters of our shapes
        if self.cellsAtEdges:
            sl_tri = self.trisize * 1.5  # Only needed if you want to put cells very close the edge of triangle chip
        else:
            sl_tri = self.trisize
        h_tri = np.sqrt(3.) / 2. * sl_tri
        # Create the triangular block
        block = Polygon([
            [-sl_tri / 2., -h_tri / 3.], [sl_tri / 2., -h_tri / 3.],
            [0, 2. * h_tri / 3.], [-sl_tri / 2., -h_tri / 3.]
        ])
        # Make a square cell
        cell = box(-cellsize / 2., -cellsize / 2., cellsize / 2., cellsize / 2.)
        # Make a lattice for the cells
        # lattice = self.createPtLattice(sl_tri, cellsize / 2. + gap / 2.,cellsize/2. + gap)
        lattice = self.createPtLattice(sl_tri, (cellsize + gap) / 2., (cellsize + gap) * np.sqrt(3.) / 2.)
        lattice = lattice + np.array(startpt)
        lattice = [
            pt for pt in lattice if Point(pt).within(block)
            ]  # Keep only points within triangular block
        # Use the lattice of points to translate the cell all over the block
        cells = [translateshape(cell, xoff=x, yoff=y) for x, y in lattice]
        # Keep only the cells that are fully within the block
        cells = [f for f in cells if f.within(block)]
        return cells

    def build_and_add_blocks(self):
        """
        Create blocks and add them to the wafer Cell
        """

        self.upCellLattice = self.getCellLattice(cellsize=2000)
        # Create a cell for the triangular blocks
        block_up = Cell('upblock')
        for x, y in self.upCellLattice:
            block_up.add(self.cells, origin=(x, y))
        # Take each point in block lattice and make a copy of the block in that location
        for x, y in self.upCenters:
            self.add(block_up, origin=(x, y))

        if self.symmetric_chips:
            for x, y in self.downCenters:
                self.add(block_up, origin=(x, y), rotation=180)
        else:
            self.downCellLattice = np.array(self.upCellLattice) * np.array([1, -1])
            block_down = Cell('downblock')
            for x, y in self.downCellLattice:
                block_down.add(self.cells, origin=(x, y))
            for x, y in self.downCenters:
                self.add(block_down, origin=(x, y))

    def plotTriangles(self, tris):
        from matplotlib import pyplot
        from matplotlib.patches import Circle
        fig = pyplot.figure(1, dpi=90)
        ax = fig.add_subplot(111)
        ax.grid()
        # Draw the wafer
        circle = Circle(
            (0, 0),
            self.wafer_r,
            facecolor="#{0:0{1}X}".format(np.random.randint(0, 16777215), 6),
            edgecolor='k',
            alpha=1)
        ax.add_patch(circle)
        tricenters = [tri.centroid.xy for tri in tris]
        x, y = zip(*tricenters)
        ax.plot(x, y, 'bo')
        # Draw all the triangles
        for i, item in enumerate(tris):
            x, y = item.exterior.coords.xy
            ax.plot(x, y, 'k-')
            patch = PolygonPatch(item,
                                 facecolor="#{0:0{1}X}".format(np.random.randint(0, 16777215), 6),
                                 edgecolor='k',
                                 alpha=0.5,
                                 zorder=2)
            ax.add_patch(patch)
        ax.axis('equal')

    def makeTriang(self, xs, ys, s, orient):
        h = np.sqrt(3.) / 2. * s
        ps = []
        for x, y in zip(xs, ys):
            if orient == "up":
                p0 = [x - s / 2., y - h / 3.]
                p1 = [x, y + 2. * h / 3.]
                p2 = [x + s / 2., y - h / 3.]
            else:
                p0 = [x - s / 2., y + h / 3.]
                p1 = [x, y - 2. * h / 3.]
                p2 = [x + s / 2., y + h / 3.]
            ps.append(Polygon([p0, p1, p2]))
        return ps

    def createPtLattice(self, size, xgap, ygap):
        G = nx.Graph(directed=False)
        G.add_node((0, 0))
        for n in xrange(int(size / min([xgap, ygap]))):
            for (q, r) in G.nodes():
                G.add_edge((q, r), (q - xgap, r - ygap))
                G.add_edge((q, r), (q + xgap, r + ygap))
                G.add_edge((q, r), (q - xgap, r + ygap))
                G.add_edge((q, r), (q + xgap, r - ygap))
        uniquepts = set(tuple(map(tuple, np.round(G.node.keys(), 10))))
        return map(np.array, uniquepts)  # Return only unique points

    def makeBlocks(self, trisize, startpt=(0, 0)):
        gap = self.block_gap
        wafer_r = self.wafer_r
        sl_tri = self.trisize  # Sidelength of the triangular blocks
        h_tri = np.sqrt(3.) / 2. * sl_tri  # Height of triangular blocks
        sl_lattice = sl_tri + gap / np.tan(
            np.deg2rad(30)
        )  # Sidelength of the block lattice (including the gaps between blocks)
        h_lattice = np.sqrt(
            3.) / 2. * sl_lattice  # Height of the lattice triangles
        # Create the lattice of the "up" facing triangles
        points = self.createPtLattice(2. * wafer_r, sl_lattice / 2., h_lattice)
        points = points + np.array([
            0, h_tri / 3. + gap / 2.
        ])  # Shift lattice so we can cleave the wafer at y=0
        points = points + np.array(startpt)  # Shift lattice by starting point if doing an MC search
        # Create the lattice of "down" facing triangles by shifting previous lattice
        points2 = points + np.array([sl_lattice / 2., h_lattice / 3])

        x, y = zip(*points)
        x2, y2 = zip(*points2)

        tris1 = self.makeTriang(np.array(x), np.array(y), sl_tri, "up")
        tris2 = self.makeTriang(np.array(x2), np.array(y2), sl_tri, "down")

        wafer = self.waferShape
        upTris = [triangle for triangle in tris1 if triangle.within(wafer)]
        downTris = [triangle for triangle in tris2 if triangle.within(wafer)]

        return upTris, downTris

    def _place_blocks(self):
        """
        Create the list of valid block sites based on block size and wafer diam.
        """
        sl_tri = self.trisize  # Sidelength of the triangular blocks
        h_tri = np.sqrt(3.) / 2. * sl_tri  # Height of triangular blocks
        if self.doMCBlockSearch:
            best = [0, 0, 0, 0]
            # Iterates many times to find the best fit
            for i in range(self.MCBlockIterations):
                # Random seed point
                #                rndpt = (np.random.randint(-sl_tri/2., sl_tri/2.), np.random.randint(-h_tri/2., h_tri/2.))
                rndpt = (0, np.random.randint(-h_tri, 0))
                # Make cells around this point
                upTris, downTris = self.makeBlocks(sl_tri, startpt=rndpt)
                NTris = (len(upTris) + len(downTris))
                if NTris > best[1]:
                    centroidDist = np.array([tri.centroid.xy for tri in upTris + downTris]).squeeze().mean(0)
                    centroidDist = np.sqrt(centroidDist[0] ** 2 + centroidDist[1] ** 2)
                    #                    centroidDist = abs(rndpt[1])
                    best = [rndpt, NTris, (upTris, downTris), centroidDist]
                elif NTris == best[1]:
                    #                    Choose the pattern that is most centered on the wafer
                    centroidDist = np.array([tri.centroid.xy for tri in upTris + downTris]).squeeze().mean(0)
                    centroidDist = np.sqrt(centroidDist[0] ** 2 + centroidDist[1] ** 2)
                    #                    centroidDist = abs(rndpt[1])
                    #                    print centroidDist
                    if centroidDist < best[3]:
                        best = [rndpt, NTris, (upTris, downTris), centroidDist]
                        #                print("Current: {:f}, Best {:f}").format(NTris,best[1])
            # Choose the best configuration (fits the most cells)
            self.upTris, self.downTris = best[2]
            self.blockOffset = best[0]
        else:
            self.upTris, self.downTris = self.makeBlocks(sl_tri)
            self.blockOffset = (0, 0)

        # Debugging
        # self.plotTriangles(self.downTris + self.upTris)
        # Find the centers of the triangles
        self.upCenters = [list(zip(*tri.centroid.xy)[0]) for tri in self.upTris]
        self.downCenters = [list(zip(*tri.centroid.xy)[0]) for tri in self.downTris]

        # %%
        sl_lattice = self.trisize + self.block_gap / np.tan(np.deg2rad(30))
        h_lattice = np.sqrt(3.) / 2. * sl_lattice
        base = h_lattice
        # Create label for each block (taken from templates._placeblocks)
        # String prefixes to associate with each row/column index
        x1s, y1s = set(), set()
        for tri in self.upTris:
            x1s.add(
                np.round(tri.centroid.x, 8)
            )  # In x use centroid as reference, in y use lower bound so up and down triangles give almost the same value
            y1s.add(base * round(float(tri.bounds[1]) / base))
        # Create dictionary of up and down triangles
        self.orientrows = dict(zip(y1s, ["up" for i, y in enumerate(y1s)]))
        # Create dictionary of up and down triangles
        x2s, y2s = set(), set()
        for tri in self.downTris:
            x2s.add(np.round(tri.centroid.x, 8))
            y2s.add(base * round(float(tri.bounds[1]) / base))
        self.orientrows.update(dict(zip(y2s, ["down" for i, y in enumerate(y2s)
                                              ])))

        x1s.update(x2s)
        xs = sorted(list(x1s))
        self.blockcols = dict(zip(xs, [
            string.uppercase[i] for i, x in enumerate(xs)
            ]))
        y1s.update(y2s)
        ys = sorted(list(y1s))
        self.blockrows = dict(zip(ys, [str(i) for i, y in enumerate(ys)]))

    # Square cell labels ex: "A", "B", "C"...
    def add_cellLabels(self, layers, center=False):
        if not (type(layers) == list): layers = [layers]
        cellLattice = sorted(self.upCellLattice,
                             key=itemgetter(1, 0))  # Sort the array first
        celllabelsUp = Cell('CellLabelsUp')
        h = self.cellsize
        vOffsetFactor = 1.
        txtSize = 200
        for i, pt in enumerate(cellLattice):
            cellid = string.uppercase[i]
            celllabel = Cell('LBL_F_' + cellid)
            for l in layers:
                txt = Label(cellid, txtSize, layer=l)
                bbox = txt.bounding_box
                offset = np.array(pt)
                txt.translate(-np.mean(bbox, 0))  # Center text around origin
                txt.translate(offset)  # Translate it to bottom of wafer
                celllabel.add(txt)
                if center:
                    celllabelsUp.add(celllabel)  # Middle of cell
                else:
                    celllabelsUp.add(celllabel, origin=(
                        0, -h / 2. * vOffsetFactor + np.mean(bbox, 0)[1]))  # Bottom of cell
        for tri in self.upTris:
            self.add(celllabelsUp, origin=tri.centroid)

        cellLattice = sorted(self.downCellLattice,
                             key=itemgetter(1, 0),
                             reverse=True)
        celllabelsDown = Cell('CellLabelsDown')
        h = self.cellsize
        for i, pt in enumerate(cellLattice):
            cellid = string.uppercase[i]
            celllabel = Cell('LBL_F_' + cellid)
            for l in layers:
                txt = Label(cellid, txtSize, layer=l)
                bbox = txt.bounding_box
                offset = np.array(pt)
                txt.translate(-np.mean(bbox, 0))  # Center text around origin
                if self.symmetric_chips:
                    txt.rotate(180)
                txt.translate(offset)  # Translate it to bottom of wafer
                celllabel.add(txt)
                if center:
                    celllabelsDown.add(celllabel)  # Middle of cell
                else:
                    celllabelsDown.add(celllabel,
                                       origin=(0, -h / 2. * vOffsetFactor + np.mean(bbox, 0)[1]))  # Bottom of cell
        for tri in self.downTris:
            self.add(celllabelsDown, origin=tri.centroid)

    # Triangular block labels ex: "A0", "B0", "C0"...
    def add_blockLabels(self, layers, center=False):
        if not (type(layers) == list): layers = [layers]
        vOffsetFactor = 1.
        blocklabelsUp = Cell('BlockLabelsUp')
        h = self.upTris[0].bounds[3] - self.upTris[0].bounds[1]
        sl_lattice = self.trisize + self.block_gap / np.tan(np.deg2rad(30))
        h_lattice = np.sqrt(3.) / 2. * sl_lattice
        base = h_lattice
        for tri in self.upTris:
            lbl_col = self.blockcols[np.round(tri.centroid.x, 8)]
            lbl_row = self.blockrows[base * round(float(tri.bounds[1]) / base)]

            blockid = str(lbl_col) + str(lbl_row)
            blocklabel = Cell('LBL_B_' + blockid)
            for l in layers:
                txt = Label(blockid, 1000, layer=l)
                bbox = txt.bounding_box
                offset = np.array(tri.centroid)
                txt.translate(-np.mean(bbox, 0))  # Center text around origin
                txt.translate(offset)  # Translate it to bottom of wafer
                blocklabel.add(txt)
                blocklabelsUp.add(blocklabel)
        if center:
            self.add(blocklabelsUp)
        else:
            self.add(blocklabelsUp, origin=(0, h / 2. * vOffsetFactor))

        blocklabelsDown = Cell('BlockLabelsDown')
        for tri in self.downTris:
            lbl_col = self.blockcols[np.round(tri.centroid.x, 8)]
            lbl_row = self.blockrows[base * round(float(tri.bounds[1]) / base)]

            blockid = str(lbl_col) + str(lbl_row)
            blocklabel = Cell('LBL_' + blockid)
            for l in layers:
                txt = Label(blockid, 1000, layer=l)
                bbox = txt.bounding_box
                offset = np.array(tri.centroid)
                txt.translate(-np.mean(bbox, 0))  # Center text around origin
                if self.symmetric_chips:
                    txt.rotate(180)
                txt.translate(offset)  # Translate it to bottom of wafer
                blocklabel.add(txt)
                blocklabelsDown.add(blocklabel)
        if center:
            self.add(blocklabelsDown)
        else:
            self.add(blocklabelsDown, origin=(0, -h / 2. * vOffsetFactor))

    def add_waferLabel(self, label, layers, pos=None):
        """
        Create a label
        """
        if not (type(layers) == list): layers = [layers]
        if self._label is None:
            self._label = Cell(self.name + '_LBL')
            self.add(self._label)
        else:
            self._label.elements = []

        offset = np.array([0, -self.wafer_r + self.block_gap]) if pos is None else np.array(pos)

        labelsize = 1000.
        for l in layers:
            txt = LineLabel(label, labelsize, style='romand', line_width=labelsize / 20., layer=l)
            bbox = txt.bounding_box
            txt.translate(-np.mean(bbox, 0))  # Center text around origin
            txt.translate(offset)  # Translate it to bottom of wafer
            self._label.add(txt)


# TODO: create a square cell helper class? Do not confuse with Cell class of gdsCAD
class Blocks(Cell):
    """
    Block object in the form of a triangle, facing either up or down
    """

    # TODO: add the inner and outer (triangle+gap) polygons to this block for easier use later
    def __init__(self, side_len, orient, name, center=[0., 0.]):
        super(Blocks, self).__init__(name)
        self.center = center
        self.orient = orient
        self.side_len = side_len
        self.height = np.sqrt(3.) / 2. * side_len
        self.ptList = self.calcPts()
        self.polygon = Polygon(self.ptList)

    def calcPts(self):
        x, y = self.center
        h = self.height
        s = self.side_len
        if self.orient == "up":
            p0 = [x - s / 2., y - h / 3.]
            p1 = [x, y + 2. * h / 3.]
            p2 = [x + s / 2., y - h / 3.]
        else:
            p0 = [x - s / 2., y + h / 3.]
            p1 = [x, y - 2. * h / 3.]
            p2 = [x + s / 2., y + h / 3.]
        ptsList = [p0, p1, p2]
        return ptsList
