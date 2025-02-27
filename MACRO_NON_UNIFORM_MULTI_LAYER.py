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

# -------------------------------
# Step 1: Create the Entire Macro
# -------------------------------
# Define key parameters
Number_large = 100           # Total length of the structure
height = 0.16                # Constant 
layer = 3
width_large = 0.46            # Large segment width (Macro)
width_small = 0.04           # Small segment width (Interlayer)
thickness_interlayer = 0.04

# Compute origin shift to center the macro
x_min = -(Number_large*(width_large+width_small))/ 2    # Leftmost edge of the rectangle
y_min = -((height+thickness_interlayer)*layer) / 2                                     # Bottom edge of the rectangle
x_max = (Number_large*(width_large+width_small))/ 2     # Rightmost edge of the rectangle
y_max = ((height+thickness_interlayer)*layer) / 2                                      # Top edge of the rectangle

# Create a new model
model_name = 'MacroModel'
part_name = 'MacroElement'
mdb.Model(name=model_name)
s = mdb.models[model_name].ConstrainedSketch(name='MacroSketch', sheetSize=20.0)

# Draw the base rectangle (centered at origin)
s.rectangle(point1=(x_min, y_min), point2=(x_max, y_max))

# Create part
p = mdb.models[model_name].Part(name=part_name, dimensionality=TWO_D_PLANAR, type=DEFORMABLE_BODY)
p.BaseShell(sketch=s)

# Delete sketch to free memory
del mdb.models[model_name].sketches['MacroSketch']

# --------------------------------------
# Step 2: Partition the Macro Element (Avoid the Final Incomplete Element)
# --------------------------------------
# Initialize counter for small segments
small_segment_count = 0  
y1 = y_min
y2 = y_min + height
for ly in range(layer):
    x_offset = x_min
    if ly != (layer-1): 
        for i in range(Number_large):  # Ensure we don't exceed 12.5 mm
            # Large segment (Macro)
            x_offset += width_large  
            p.PartitionFaceByShortestPath(
                faces=p.faces, 
                point1=(x_offset, y1, 0.0),  
                point2=(x_offset, y2, 0.0)
            )

            # Small segment (Interlayer)
            if x_offset + width_small <= x_max:  # Only partition if it fits exactly
                if i == Number_large-1:
                    continue
                x_offset += width_small  
                p.PartitionFaceByShortestPath(
                    faces=p.faces, 
                    point1=(x_offset, y1, 0.0),  
                    point2=(x_offset, y2, 0.0)
                )
                small_segment_count += 1  # Increment counter
    else:
          for i in range(Number_large):  # Ensure we don't exceed 12.5 mm
            # Large segment (Macro)
            x_offset += width_large  
            p.PartitionFaceByShortestPath(
                faces=p.faces, 
                point1=(x_offset, y1, 0.0),  
                point2=(x_offset, y2+thickness_interlayer, 0.0)
            )

            # Small segment (Interlayer)
            if x_offset + width_small <= x_max:  # Only partition if it fits exactly
                if i == Number_large-1:
                    continue
                x_offset += width_small  
                p.PartitionFaceByShortestPath(
                    faces=p.faces, 
                    point1=(x_offset, y1, 0.0),  
                    point2=(x_offset, y2+thickness_interlayer, 0.0)
                )
                small_segment_count += 1  # Increment counter
    p.PartitionFaceByShortestPath(
            faces=p.faces, 
            point1=(x_min, y2, 0.0),  
            point2=(x_max, y2, 0.0)
        )
    if ly != layer-1:
        p.PartitionFaceByShortestPath(
                faces=p.faces, 
                point1=(x_min, y2+thickness_interlayer, 0.0),  
                point2=(x_max, y2+thickness_interlayer, 0.0)
            )
    y1 += (height + thickness_interlayer) 
    y2 += (height + thickness_interlayer)
    

# Create materials
mdb.models[model_name].Material(name='Material_Macro')
mdb.models[model_name].Material(name='Material_Interlayer')

# Create sections
mdb.models[model_name].HomogeneousSolidSection(name='Macro', material='Material_Macro', thickness=None)
mdb.models[model_name].HomogeneousSolidSection(name='Interlayer', material='Material_Interlayer', thickness=None)

# --------------------------------------
# Step 3: Use getSequenceFromMask() to Select Large and Small Macro Sets
# --------------------------------------

# Assign faces to sets
x_offset = x_min 

# Store center positions for small segments (Interlayer)
positions_interlayer = []
positions_macro = []
positions_interlayer_y = []

# Compute fiber center positions for small segments (Interlayer)
y = y_min + (height/2)
for _ in range(layer):
    x = x_offset + width_large + (width_small / 2)
    for i in range(Number_large):
        positions_interlayer.append((x, y, 0))  # Store (x, y, z) position
        x += (width_large + width_small)   # Move to next large segment
    y += (height + thickness_interlayer)
# Compute center positions for macro sections
y = y_min + (height/2)
for _ in range(layer):
    x = x_offset + (width_large / 2)
    for i in range(Number_large): # One extra macro section at the end
        positions_macro.append((x, y, 0))  # Store (x, y, z) position
        x += (width_large + width_small)   # Move to next large segment
    y += (height + thickness_interlayer)

y = y_min + (height + (thickness_interlayer/2))
for ly in range(layer-1):
    positions_interlayer_y.append((0, y, 0))
    y += (height + thickness_interlayer)
    if ly == layer-1:
        x = x_offset + (width_large / 2)
        for i in range(Number_large*2): # One extra macro section at the end
            positions_interlayer.append((x, y_max-(thickness_interlayer/2), 0))  # Store (x, y, z) position
            x += (width_large/2 + width_small/2) 

# Assign Interlayer Section to closest faces
for point in positions_interlayer:
    closest_face = p.faces.getClosest(coordinates=(point,))[0][0]  # Find the closest face
    p.SectionAssignment(
        offset=0.2, 
        offsetField='', 
        offsetType=MIDDLE_SURFACE, 
        region=Region(faces=p.faces[closest_face.index:closest_face.index+1]), 
        sectionName='Interlayer', 
        thicknessAssignment=FROM_SECTION
    )
# Assign Macro Section to closest faces
for point in positions_macro:
    closest_face = p.faces.getClosest(coordinates=(point,))[0][0]  # Find the closest face
    p.SectionAssignment(
        offset=0.2, 
        offsetField='', 
        offsetType=MIDDLE_SURFACE, 
        region=Region(faces=p.faces[closest_face.index:closest_face.index+1]), 
        sectionName='Macro', 
        thicknessAssignment=FROM_SECTION
    )

for point in positions_interlayer_y:
    closest_face = p.faces.getClosest(coordinates=(point,))[0][0]  # Find the closest face
    p.SectionAssignment(
        offset=0.2, 
        offsetField='', 
        offsetType=MIDDLE_SURFACE, 
        region=Region(faces=p.faces[closest_face.index:closest_face.index+1]), 
        sectionName='Interlayer', 
        thicknessAssignment=FROM_SECTION
    )


