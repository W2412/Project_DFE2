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
def generate_non_overlapping_rectangles(s, x_bounds, y_bounds, height_range,Vf):
    """Generates non-overlapping rectangles."""
    existing_rectangles = []
    centers = []
    total_area = 0
    total_domain_area = (x_bounds[1] - x_bounds[0]) * (y_bounds[1] - y_bounds[0])  # Domain area
    l = 0
    while total_area < Vf * total_domain_area:
        attempts = 0
        while attempts < 100:  # Limit to avoid infinite loops
            width = random.uniform(0.00706, 0.06001)
            height = height_range
            orientation = random.randint(0,1)
            if  orientation == 1:
                x1 = -0.0353
                y1 = random.uniform(y_bounds[0], y_bounds[1])
                x2 = x1 + width
                y2 = y1 + height
            else:
                x2 = 0.0353
                y2 = random.uniform(y_bounds[0], y_bounds[1])
                x1 = x2 - width
                y1 = y2 - height
            # Ensure rectangle is within bounds
            if not (x_bounds[0] <= x1 and x2 <= x_bounds[1] and y_bounds[0] + gap <= y1 and y2 <= y_bounds[1] - gap):
                attempts += 1
                continue
            
            # Check if the new rectangle overlaps any existing one
            if not (any((x1 < ex[2] and x2 > ex[0] and y1 < ex[3] + gap and y2  > ex[1] - gap) for ex in existing_rectangles)):
                # Add the rectangle to the sketch-
                remaining_area = (Vf * total_domain_area) - total_area
                rectangle_area = width * height

                if rectangle_area > remaining_area:
                    width = remaining_area / height  # Adjust width to exactly match the remaining area
                    if orientation == 1:
                        x2 = x1 + width 
                    else:
                        x1 = x2 - width
                    rectangle_area = width * height  # Recalculate area
                
                s.rectangle(point1=(x1, y1), point2=(x2, y2))
                existing_rectangles.append((x1, y1, x2, y2))
                centers.append((((x1+x2)/2),(y1+y2)/2))
                total_area += rectangle_area
                break
            attempts += 1
    return centers


# Define bounds and generate rectangles
Vf = 0.25
x_bounds = (-0.0353, 0.0353)  # X-coordinate range
y_bounds = (-0.0353, 0.0353)  # Y-coordinate range
height_range =0.0095 # Height range of rectangles
gap = 0.002325 
x = generate_non_overlapping_rectangles(s, x_bounds, y_bounds, height_range,Vf)

# Name the part model and associate it
p = mdb.models['Model-1'].Part(name='RVE2DFibre', dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p = mdb.models['Model-1'].parts['RVE2DFibre']

if len(s.geometry) == 0:
    print("Error: The sketch is empty!")

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
session.viewports['Viewport: 1'].view.setValues(width=0.1, height=0.1)

# Sketch RVE Rectangle
s1.rectangle(point1=(-0.0353, -0.0353), point2=(0.0353, 0.0353))

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
        faces=mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].faces.findAt(((-0.03,-0.03 , 0.0),))
    ), 
    sectionName='matrixSection', 
    thicknessAssignment=FROM_SECTION
)

# Assign sections to fibers
for i in range(len(x)):
    mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].SectionAssignment(
        offset=0.2, 
        offsetField='', 
        offsetType=MIDDLE_SURFACE, 
        region=Region(
            faces=mdb.models['Aligned_ShortFibreComposite'].parts['RVE2DComposite'].faces.findAt(((x[i][0], x[i][1], 0.0), (0.0, 0.0, 1.0)))
        ), 
        sectionName='fibreSection', 
        thicknessAssignment=FROM_SECTION
    )
#************************************************************************
#                               END OF SCRIPT                             
#************************************************************************
