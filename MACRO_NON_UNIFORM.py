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
Number_large = 20     # Total length of the structure
height = 0.2          # Constant height
width_large = 0.6     # Large segment width (Macro)
width_small = 0.04    # Small segment width (Interlayer)

# Compute origin shift to center the macro
x_min = -(Number_large*(width_large+width_small))/ 2    # Leftmost edge of the rectangle
y_min = -height / 2                                     # Bottom edge of the rectangle
x_max = (Number_large*(width_large+width_small))/ 2     # Rightmost edge of the rectangle
y_max = height / 2                                      # Top edge of the rectangle

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
x_offset = x_min  # Start from the leftmost edge
# Initialize counter for small segments
small_segment_count = 0  

for i in range(Number_large):  # Ensure we don't exceed 12.5 mm
    # Large segment (Macro)
    x_offset += width_large  
    p.PartitionFaceByShortestPath(
        faces=p.faces, 
        point1=(x_offset, y_min, 0.0),  
        point2=(x_offset, y_max, 0.0)
    )

    # Small segment (Interlayer)
    if x_offset + width_small <= x_max:  # Only partition if it fits exactly
        if i == 19:
            continue
        x_offset += width_small  
        p.PartitionFaceByShortestPath(
            faces=p.faces, 
            point1=(x_offset, y_min, 0.0),  
            point2=(x_offset, y_max, 0.0)
        )
        small_segment_count += 1  # Increment counter


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
x_positions_interlayer = []
x_positions_macro = []

# Compute fiber center positions for small segments (Interlayer)
x = x_offset + width_large + (width_small / 2)
for i in range(small_segment_count + 1):
    x_positions_interlayer.append((x, 0, 0))  # Store (x, y, z) position
    x += (width_large + width_small)   # Move to next large segment

# Compute center positions for macro sections
x = x_offset + (width_large / 2)
for i in range(small_segment_count + 1):  # One extra macro section at the end
    x_positions_macro.append((x, 0, 0))  # Store (x, y, z) position
    x += (width_large + width_small)   # Move to next large segment

# Debugging: Print positions before assignment
print("Total Small Segments (Interlayer):", len(x_positions_interlayer))
print("Total Large Segments (Macro):", len(x_positions_macro))

# Assign Interlayer Section to closest faces
for point in x_positions_interlayer:
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
for point in x_positions_macro:
    closest_face = p.faces.getClosest(coordinates=(point,))[0][0]  # Find the closest face
    p.SectionAssignment(
        offset=0.2, 
        offsetField='', 
        offsetType=MIDDLE_SURFACE, 
        region=Region(faces=p.faces[closest_face.index:closest_face.index+1]), 
        sectionName='Macro', 
        thicknessAssignment=FROM_SECTION
    )


# # --------------------------------------
# # Step 5: Create and Assign Mesh
# # --------------------------------------
# # Define mesh size
# mesh_size = 0.02  # Set appropriate element size

# # Seed the part with defined mesh size
# p.seedPart(size=mesh_size, deviationFactor=0.1, minSizeFactor=0.1)

# # Assign mesh controls
# p.setMeshControls(regions=p.faces, elemShape=QUAD, technique=STRUCTURED)

# # Generate mesh
# p.generateMesh()

