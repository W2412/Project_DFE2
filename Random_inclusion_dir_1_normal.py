import random

#******************************************************************************************************** 
# RVE2DFibre Creation Script with Random Fiber Placement
# Author: Modified Version by Assistant
# Date: 08-Dec-2024
# Purpose: Creates Abaqus model (RVE2DFibre) with random fiber placements.
#******************************************************************************************************** 

# Import Abaqus-related (Python) Object files
from abaqus import *
from abaqusConstants import *
from regionToolset import Region
import __main__
import section 
import regionToolset
import displayGroupMdbToolset as dgm
import part
import material
import assembly
import step
import interaction
import load
import mesh
import job
import sketch
import visualization
import xyPlot
import displayGroupOdbToolset as dgo
import connectorBehavior

#****************************************************
# CREATE MATRIX AND FIBRE MATERIALS/SECTIONS HERE
#****************************************************
mdb.models['Model-1'].Material(name='EPOXY')
mdb.models['Model-1'].Material(name='EGLASS')
mdb.models['Model-1'].HomogeneousSolidSection(name='matrixSection', material='EPOXY', thickness=None)
mdb.models['Model-1'].HomogeneousSolidSection(name='fibreSection', material='EGLASS', thickness=None)

# Create Viewport
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(meshTechnique=OFF)

s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2160)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=STANDALONE)
session.viewports['Viewport: 1'].view.setValues(width=0.5, height=0.5)

#****************************************************
# RANDOMLY GENERATE RECTANGULAR INCLUSIONS
#****************************************************
def generate_non_overlapping_rectangles(s, num_rectangles, x_bounds, y_bounds, width_range, height_range, gap):
    """Generates non-overlapping rectangles with a specified gap."""
    existing_rectangles = []
    centers = []

    for _ in range(num_rectangles):
        attempts = 0
        while attempts < 100:  # Limit to avoid infinite loops
            x1 = random.uniform(x_bounds[0] , x_bounds[1])
            y1 = random.uniform(y_bounds[0] , y_bounds[1])
            print(x1,y1)
            x2 = x1 + width_range
            y2 = y1 + height_range

            # Ensure rectangle is within bounds
            if not ((x_bounds[0] + gap) <= x1 and x2 <= (x_bounds[1] - gap) and (y_bounds[0] + gap) <= y1 and y2 <= (y_bounds[1] - gap)):
                attempts += 1
                continue

            # Check for overlap and enforce gap
            if not any(
                (x1 - gap < ex[2] and x2 + gap > ex[0] and y1 - gap < ex[3] and y2 + gap > ex[1])
                for ex in existing_rectangles
            ):
                # Add the rectangle to the sketch
                s.rectangle(point1=(x1, y1), point2=(x2, y2))
                existing_rectangles.append((x1, y1, x2, y2))
                centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
                break

            attempts += 1

    return centers


# Define bounds and generate rectangles
num_rectangles = 5
x_bounds = (-0.095, 0.095)  # X-coordinate range
y_bounds = (-0.095, 0.095)  # Y-coordinate range
width_range = 0.136 # Width range of rectangles
height_range = 0.0095 # Height range of rectangles
gap = 0.002375 
x = generate_non_overlapping_rectangles(s, num_rectangles, x_bounds, y_bounds, width_range, height_range, gap)
session.viewports['Viewport: 1'].setValues(displayedObject=s)

# Name the part model and associate it
p = mdb.models['Model-1'].Part(name='RVE2DFibre', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['RVE2DFibre']

# Fibre Extrusion
p.BaseShell(sketch=s)
s.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['RVE2DFibre']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']

#****************************************************
# MATRIX SECTION
#****************************************************

# Create Viewport
session.viewports['Viewport: 1'].partDisplay.setValues(mesh=OFF)
session.viewports['Viewport: 1'].partDisplay.meshOptions.setValues(meshTechnique=OFF)

s1 = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2160)
g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
s1.setPrimaryObject(option=STANDALONE)
session.viewports['Viewport: 1'].view.setValues(width=0.5, height=0.5)

# Sketch RVE Rectangle
s1.rectangle(point1=(-0.095, -0.095), point2=(0.095, 10.095))

# Name the part model and associate it
p = mdb.models['Model-1'].Part(name='RVE2DMatrix', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['RVE2DMatrix']

# Matrix Extrusion
p.BaseShell(sketch=s1)
s1.unsetPrimaryObject()
p = mdb.models['Model-1'].parts['RVE2DMatrix']
session.viewports['Viewport: 1'].setValues(displayedObject=p)
del mdb.models['Model-1'].sketches['__profile__']

#****************************************************
# ASSEMBLY INSTANCES AND MERGE THE TWO INSTANCES
#****************************************************
a = mdb.models['Model-1'].rootAssembly
a.DatumCsysByDefault(CARTESIAN)
p = mdb.models['Model-1'].parts['RVE2DFibre']
a.Instance(name='RVE2DFibre-1', part=p, dependent=ON)
p = mdb.models['Model-1'].parts['RVE2DMatrix']
a.Instance(name='RVE2DMatrix-1', part=p, dependent=ON)
a = mdb.models['Model-1'].rootAssembly
a.InstanceFromBooleanMerge(name='RVE2DComposite', instances=(
    a.instances['RVE2DFibre-1'], a.instances['RVE2DMatrix-1'], ),
    keepIntersections=ON, originalInstances=SUPPRESS, domain=GEOMETRY)
mdb.models['Model-1'].rootAssembly.features.changeKey(
    fromName='RVE2DComposite-1', toName='RVE2DComposite')

#****************************************************
# EXTRUDE-CUT SECTION TO TRIM BOUNDARY FIBRES
#****************************************************
session.viewports['Viewport: 1'].partDisplay.setValues(sectionAssignments=OFF, engineeringFeatures=OFF)
p1 = mdb.models['Model-1'].parts['RVE2DComposite']
session.viewports['Viewport: 1'].setValues(displayedObject=p1)
s = mdb.models['Model-1'].ConstrainedSketch(name='__profile__', sheetSize=2160)
g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
s.setPrimaryObject(option=SUPERIMPOSE)
p = mdb.models['Model-1'].parts['RVE2DComposite']
p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
s.rectangle(point1=(-0.095, -0.095), point2=(0.095, 0.095))
s.rectangle(point1=(-1440, -400), point2=(1440, 400))
session.viewports['Viewport: 1'].view.fitView()
p = mdb.models['Model-1'].parts['RVE2DComposite']
p.Cut(sketch=s)
s.unsetPrimaryObject()
mdb.models.changeKey(fromName='Model-1', toName='Aligned_ShortFibreComposite')
session.viewports['Viewport: 1'].setValues(displayedObject=None)

# Assign section to the matrix
mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].SectionAssignment(
    offset=0.0, 
    offsetField='', 
    offsetType=MIDDLE_SURFACE, 
    region=Region(
        faces=mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].faces.findAt(((0.00475 , 0 , 0.0),))
    ), 
    sectionName='matrixSection', 
    thicknessAssignment=FROM_SECTION
)

# Assign sections to fibers
# Assign sections to fibers
for i in range(num_rectangles):
    point = (x[i][0], x[i][1], 0.0)  # Fiber center coordinates
    closest_face = mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].faces.getClosest(coordinates=(point,))[0][0]
    mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].SectionAssignment(
        offset=0.2, 
        offsetField='', 
        offsetType=MIDDLE_SURFACE, 
        region=Region(faces=mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].faces[closest_face.index:closest_face.index+1]), 
        sectionName='fibreSection', 
        thicknessAssignment=FROM_SECTION)

#************************************************************************
#                               END OF SCRIPT                             
#************************************************************************
