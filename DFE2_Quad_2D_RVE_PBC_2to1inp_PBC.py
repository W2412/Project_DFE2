    # -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:07:06 2023

@author: K. M. Yeoh
"""

import os
import numpy as np
import math

### Update the following parameters
os.chdir('C:\\temp\\DirectFE2\\DFE2_FOLDER') # directory where input files are, use double '\'
MacroInpName = 'MACRO_FINAL_Coupled_thermal.inp' # Name of macroscale input file
RVEInpName = 'RVE_CF_PEEK_23_020_Coupled_thermal.inp' # Name of RVE input file 
NewInpName = 'DFE2_2D_ALPHA2_CF_PEEK.inp' # Name of new Direct FE2 input file

GP = [[-3**-0.5,-3**-0.5],[3**-0.5,-3**-0.5],[3**-0.5,3**-0.5],[-3**-0.5,3**-0.5]]

### Define functions
# Search inp array for a particular part's elements and nodes
def Search(inp,key1,out1,type1): #if dealine with nodes and coordinates, set type1 to 1 
    for i in range(len(inp)):
        if inp[i].count(key1)!=0:
            break
    for temp_line in inp[i+1:]:
        if (temp_line == '') or (temp_line.count("*") != 0):
            break
        temp_line = (temp_line.replace(',',' ')).split()
        temp_line.pop(0) #Removes element number in connectivity and node number in nodal coordinates
        for i in range(len(temp_line)):
            if type1 == 1:
                temp_line[i] = float(temp_line[i])
            else:
                temp_line[i] = int(temp_line[i])-1 #All node/element labels are -1 for use in Python as indices
        out1.append(temp_line)  

# Removes corner nodes from an edge
def TakeVertexOut(face):
    face.pop(0)
    face.pop(-1)
    return face  

# Calculates bilinear shape function values
def Bilin_Interpolation(tsi,eta):
    N1=float(0.25*(1-tsi)*(1-eta))
    N2=float(0.25*(1+tsi)*(1-eta))
    N3=float(0.25*(1+tsi)*(1+eta))
    N4=float(0.25*(1-tsi)*(1+eta))
    return [N1,N2,N3,N4]

# Sorts nodes along an edge using their coordinates
def SortListofNodes1D(faceN,coordinate): # 0 for x; 1 for y; 2 for z; gives node numbers ready to be called with python (already -1)
    newlist = []
    oldlist = []
    for i in range(len(faceN)):
        oldlist.append(RVENodalCoord[faceN[i]][coordinate])
    
    orderedlist = sorted(oldlist)
    for j in range(len(orderedlist)):
        ind = oldlist.index(orderedlist[j])
        newlist.append(faceN[ind])
    
    return newlist

def distance_from_origin(points):
    min_distance = float('inf')  # Initialize with a very large value
    nearest_point = None
    
    for i in range(len(points)):  # Iterate through rows of the 2D array
        distance = math.sqrt(points[i][0]**2 + points[i][1]**2)  # Compute distance
        if distance < min_distance:  # Update nearest point if distance is smaller
            min_distance = distance
            nearest_point = points[i]
            node = i
    
    return nearest_point, node   # Return the point nearest to (0, 0)

### Extracting information from Macro and RVE input files
# Macroscale input file
inp1 = []    
f1 = open(MacroInpName,'r')

while 1:
    line = f1.readline()
    if not line:
        break
    line = line.strip() #removes additional white spaces on left and right
    inp1.append(line)
f1.close()
file_name = "output.txt"
with open(file_name, 'w') as file:
    for item in inp1:
        file.write(f"{item}\n")

# Removing 'generate' for easier processing
for i in reversed(range(len(inp1))):
    num = 0
    if (inp1[i].count('generate')!=0):
        num += 1
        # print(i)
        # print(inp1[i])
        # print(inp1[378])
        Temp = (inp1[i+1].replace(',',' ')).split()
        # print(Temp)
        k = 0 #term counter
        m = 0 #extra line counter
        for j in range(int(Temp[0]),int(Temp[1])+1,int(Temp[2])):
            if k==0:
                Temp2 = str(j)
                k = k+1
            elif k==16:
                inp1.insert(i+2+m,Temp2)
                # print(Temp2)
                m = m+1
                Temp2 = str(j)
                k = 1
            else:
                Temp2 = Temp2+', '+str(j)
                k = k+1
        inp1.insert(i+2+m,Temp2)
        inp1[i] = inp1[i][0:len(inp1[i])-10]
for i in reversed(range(len(inp1))): 
    if (inp1[i].count('*Nset') or inp1[i].count('*Elset')!=0):
        print(inp1[i+1])
        inp1.remove(inp1[i+1])


file_name = "output2.txt"
with open(file_name, 'w') as file:
    for item in inp1:
        file.write(f"{item}\n")

# RVE input file
inp2 = []    
f2 = open(RVEInpName,'r')

while 1:
    line = f2.readline()
    if not line:
        break
    line = line.strip() #removes additional white spaces on left and right
    inp2.append(line)
    
f2.close()

# Removing 'generate' for easier processing
for i in reversed(range(len(inp2))):
    if (inp2[i].count('generate')!=0):
        Temp = (inp2[i+1].replace(',',' ')).split()
        k = 0 #term counter
        m = 0 #extra line counter
        for j in range(int(Temp[0]),int(Temp[1])+1,int(Temp[2])):
            if k==0:
                Temp2 = str(j)
                k = k+1
            elif k==16:
                inp2.insert(i+2+m,Temp2)
                m = m+1
                Temp2 = str(j)
                k = 1
            else:
                Temp2 = Temp2+', '+str(j)
                k = k+1
        inp2.insert(i+2+m,Temp2)
        inp2[i] = inp2[i][0:len(inp2[i])-10]
        inp2.remove(inp2[i+1])

# Extracting macroscale element info from old inp file
MacroNodalConnect,MacroNodalCoord = [],[]
Search(inp1,'*Element',MacroNodalConnect,0)
Search(inp1,'*Node',MacroNodalCoord,1)


StartConst = 'a'
for i in range(len(inp1)):
    if (inp1[i].count('*Instance,'))!=0:
        Line = inp1[i].split(', ')
        MacroInstName = Line[1][5:]
    if (inp1[i].count('*Material,'))!=0:
        inp1[i+2] = '1e-10,1e-10'
    if (inp1[i].count('** Section'))!=0:
        if inp1[i+2] == ',':
            Thickness = 1.0
        else:
            Thickness = float(inp1[i+2].strip(','))
    if (inp1[i].count('** Constraint'))!=0 and (StartConst == 'a'):
        StartConst = i

# Extracting RVE info from old inp file
RVENodalCoord = []
Search(inp2,'*Node',RVENodalCoord,1)

Sections = []
Materials = []
StartEle = 'a'
for i in range(len(inp2)):
    if (inp2[i].count('*Element'))!=0 and (StartEle == 'a'):
        StartEle = i
    if (inp2[i].count('** Section'))!=0:
        Sections.append(i)
    if (inp2[i].count('*Material'))!=0:
        Materials.append(i)
RVEMats = open('RVEMats.dat','w')
for i in range(len(Materials)):
    for j in range(Materials[i]+1,len(inp2)):
        if (inp2[j].count('*Material'))!=0 or (inp2[j].count('**'))!=0:
            MatEnd = j
            break
        MatEnd = j # If no further information provided beyond RVE materials
    if i == len(Materials)-1:
        for j in range(Materials[i],MatEnd+1):
            print(inp2[j],file=RVEMats)
    else:
        for j in range(Materials[i],MatEnd):
            print(inp2[j],file=RVEMats)
RVEMats.close()


### Processing the macroscale part information
# Sorting the nodal connectivity to match with DFE2 conventions


# print(MacroNodalConnect )
# print("\n --------")
# print(MacroNodalCoord)
Nele = len(MacroNodalConnect)
NodalConnect = []
NodalCoordX = []
NodalCoordY = []
for i in range(Nele):
    Nodes = [] #Temp Nodes
    X = []
    Y = []
    Ang = []
    TempConnect = []
    TempCoordX = []
    TempCoordY = []
    
    # Obtaining nodal information from this particular element
    for j in range(4):
        Nodes.append(MacroNodalConnect[i][j])
        X.append(MacroNodalCoord[MacroNodalConnect[i][j]][0])
        Y.append(MacroNodalCoord[MacroNodalConnect[i][j]][1])
    # Obtaining centroid of the regtangular element
    X0 = sum(X)/4
    Y0 = sum(Y)/4
    # Obtaining orientation of each node relative to centroid
    for j in range(4):
        X1 = X[j]-X0
        Y1 = Y[j]-Y0
        theta = math.atan(Y1/X1)
        if X1<0:
            theta = theta+math.pi
        if theta<0:
            theta = theta+2*(math.pi)
        Ang.append(theta*360/(2*(math.pi)))
    SAng = sorted(Ang)
    
    for j in range(4):
        Order1 = [2,3,0,1]
        N = Ang.index(SAng[Order1[j]])
        TempConnect.append(Nodes[N])
        TempCoordX.append(X[N])
        TempCoordY.append(Y[N])
    NodalConnect.append(TempConnect)
    NodalCoordX.append(TempCoordX)
    NodalCoordY.append(TempCoordY)


### Processing the RVE part information
# Finding the smallest and largest nodal coordinate in both directions
RVE_ListX = []
RVE_ListY = []
for i in range(len(RVENodalCoord)):
    RVE_ListX.append(RVENodalCoord[i][0])
    RVE_ListY.append(RVENodalCoord[i][1])
xMin = min(RVE_ListX)
xMax = max(RVE_ListX)
yMin = min(RVE_ListY)
yMax = max(RVE_ListY)
del RVE_ListX
del RVE_ListY

# Sorting the RVE boundary nodes
FaceLNodes,FaceRNodes,FaceBNodes,FaceTNodes = [],[],[],[]
for i in range(len(RVENodalCoord)):
    if RVENodalCoord[i][0] == xMin:
        FaceLNodes.append(i)
    if RVENodalCoord[i][0] == xMax:
        FaceRNodes.append(i)
    if RVENodalCoord[i][1] == yMin:
        FaceBNodes.append(i)
    if RVENodalCoord[i][1] == yMax:
        FaceTNodes.append(i)
for i in range(len(FaceLNodes)):
    if FaceLNodes[i] in FaceBNodes:
        V1 = FaceLNodes[i]
    if FaceLNodes[i] in FaceTNodes:
        V4 = FaceLNodes[i]
for i in range(len(FaceRNodes)):
    if FaceRNodes[i] in FaceBNodes:
        V2 = FaceRNodes[i]
    if FaceRNodes[i] in FaceTNodes:
        V3 = FaceRNodes[i]
# print(FaceLNodes)
# print(FaceRNodes)
# print(FaceTNodes)
# print(FaceBNodes)
print(SortListofNodes1D(FaceLNodes,1))
    
# Sorting RVE dimensions and offsets
B_RVE = xMax - xMin
H_RVE = yMax - yMin
OffsetX = (xMax + xMin)/2
OffsetY = (yMax + yMin)/2
Offset = [OffsetX,OffsetY]

# Adjusting RVE nodal coordinates to correspond to a part centered at the origin
for i in RVENodalCoord:
    for j in range(2):
        i[j] = i[j] - Offset[j]
        

### Generating the RVE placement in the macroscale mesh
RVEParts = open('RVEParts.dat','w')
Insts = open('Insts.dat','w')
SF = []
for i in range(Nele):
    C = np.array([[1,-1,-1,1],[1,1,-1,-1],[1,1,1,1],[1,-1,1,-1]])
    C_inv = np.linalg.inv(C)
    [a0,a1,a2,a3] = np.dot(C_inv,NodalCoordX[i])
    [b0,b1,b2,b3] = np.dot(C_inv,NodalCoordY[i])

    for j in range(4):
        [tsi,eta] = GP[j]
        J = np.array([[a1+a3*eta,b1+b3*eta],[a2+a3*tsi,b2+b3*tsi]])
        J_RVE = abs(np.linalg.det(J)/(B_RVE*H_RVE)) # Scaling factor for thicknes
        
        [N1,N2,N3,N4] = Bilin_Interpolation(tsi,eta)
        RVE_X = N1*NodalCoordX[i][0] + N2*NodalCoordX[i][1] + N3*NodalCoordX[i][2] + N4*NodalCoordX[i][3]
        RVE_Y = N1*NodalCoordY[i][0] + N2*NodalCoordY[i][1] + N3*NodalCoordY[i][2] + N4*NodalCoordY[i][3]
        
        if round(J_RVE,5) in SF: # If such an RVE was created previously, reuse it for the next instance
            N = SF.index(round(J_RVE,5))
        else: # If not, then we create another RVE with this current thickness scaling
            N = len(SF)
            SF.append(round(J_RVE,5))
            print('**',file=RVEParts)
            print('*Part, name=RVE-'+str(N+1),file=RVEParts)
            print('*Node',file=RVEParts)
            for k in range(len(RVENodalCoord)):
                print(str(k+1)+', '+str(RVENodalCoord[k][0])+', '+str(RVENodalCoord[k][1]),file=RVEParts)
                
            for k in range(StartEle,Sections[0]):
                print(inp2[k],file=RVEParts)
                
            for k in range(len(Sections)):
                print(inp2[Sections[k]]+'-'+str(N+1),file=RVEParts)
                print(inp2[Sections[k]+1],file=RVEParts)
                if inp2[Sections[k]+1].count('*Solid Section')!=0:
                    print(str(Thickness*J_RVE)+',',file=RVEParts)
                elif inp2[Sections[k]+1].count('*Cohesive Section')!=0:
                    Line = inp2[Sections[k]+2].split(',')
                    print(str(Line[0])+','+str(Thickness*J_RVE),file=RVEParts)
                
            print('*End Part',file=RVEParts)
            
        print('*Instance, name=Ele'+str(i+1)+'-RVE'+str(j+1)+', part=RVE-'+str(N+1),file=Insts)
        print(str(RVE_X)+', '+str(RVE_Y)+', 0.',file=Insts)
        print('*End Instance',file=Insts)
        print('**',file=Insts)
        # print(RVE_X,RVE_Y)
            
RVEParts.close()
Insts.close()
# print(C_inv)

            
### Setting up the MPCs
Sets = open('Sets.dat','w')
Eqns = open('Eqns.dat','w')

# Pairing the nodes
FaceLNodeswithV = FaceLNodes
FaceRNodeswithV = FaceRNodes
FaceLNodes = TakeVertexOut(SortListofNodes1D(FaceLNodes,1))
FaceRNodes = TakeVertexOut(SortListofNodes1D(FaceRNodes,1))
PairingFacesLR = []
for i in range(len(FaceLNodes)):
    Temp = []
    Temp.append(FaceLNodes[i])
    Temp.append(FaceRNodes[i])
    PairingFacesLR.append(Temp)

FaceBNodeswithV = FaceBNodes
FaceTNodeswithV = FaceTNodes
FaceBNodes = TakeVertexOut(SortListofNodes1D(FaceBNodes,0))
FaceTNodes = TakeVertexOut(SortListofNodes1D(FaceTNodes,0))
PairingFacesBT = []
for i in range(len(FaceBNodes)):
    Temp = []
    Temp.append(FaceBNodes[i])
    Temp.append(FaceTNodes[i])
    PairingFacesBT.append(Temp)

Nodallist = []
dummy = []
for i in range(len(RVENodalCoord)):
    Nodallist.append(i)
    dummy.append(i)

number = len(Nodallist)
for i in range(number):
    if i in FaceLNodeswithV:
        Nodallist.remove(i)
    elif i in FaceRNodeswithV:
        Nodallist.remove(i)
    elif i in FaceBNodeswithV:
        Nodallist.remove(i)
    elif i in FaceTNodeswithV:
        Nodallist.remove(i)

for i in Nodallist:
    if i in range(number):
        dummy.remove(i)

center, center_node  = distance_from_origin(RVENodalCoord)


edgeR = FaceRNodeswithV[int(len(FaceRNodeswithV))//2-1]
edgeL = FaceLNodeswithV[int(len(FaceLNodeswithV))//2-1]
edgeT = FaceTNodeswithV[int(len(FaceTNodeswithV))//2-1]
edgeB = FaceBNodeswithV[int(len(FaceBNodeswithV))//2-1]
# print(dummy)
# print(Nodallist)
    
# Calculating the coefficients and setting up the MPCs
for i in range(Nele):
    C = np.array([[1,-1,-1,1],[1,1,-1,-1],[1,1,1,1],[1,-1,1,-1]])
    C_inv = np.linalg.inv(C)
    [a0,a1,a2,a3] = np.dot(C_inv,NodalCoordX[i])
    [b0,b1,b2,b3] = np.dot(C_inv,NodalCoordY[i])
    
    # Calling macroscale nodes into sets
    for j in range(4):
        print('*Nset, nset=Ele'+str(i+1)+'-N'+str(j+1)+', instance='+str(MacroInstName),file=Sets)
        print(str(NodalConnect[i][j]+1),file=Sets)
    
    # At each macroscale GP
    for j in range(4): # 4 GPs
        [tsi,eta] = GP[j]
        
        J = np.array([[a1+a3*eta,b1+b3*eta],[a2+a3*tsi,b2+b3*tsi]])
        J_inv = np.linalg.inv(J)
        dN1 = [-0.25*(1-eta),-0.25*(1-tsi)]
        dN2 = [0.25*(1-eta),-0.25*(1+tsi)]
        dN3 = [0.25*(1+eta),0.25*(1+tsi)]
        dN4 = [-0.25*(1+eta),0.25*(1-tsi)]
        N_NatDeriv = [dN1,dN2,dN3,dN4]
        N_GloDeriv = [[],[],[],[]]
        Shape_fn = Bilin_Interpolation(tsi,eta)
        
        # Calculating shape function gradients
        for k in range(4):
            N_GloDeriv[k] = np.dot(J_inv,np.transpose(np.array(N_NatDeriv[k])))

        # Calling sets and setting up the MPCs for left and right faces
        for k in range(len(PairingFacesLR)):
            print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeL'+str(k+1)+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
            print(str(PairingFacesLR[k][0]+1),file=Sets)
            print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeR'+str(k+1)+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
            print(str(PairingFacesLR[k][1]+1),file=Sets)
            
            for dof in [0, 1, 10]:
                print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-LR'+str(k+1)+'-DOF'+str(dof+1),file=Eqns)
                print('*Equation',file=Eqns)
                print('6',file=Eqns)
                print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeR'+str(k+1)+', '+str(dof+1)+', -1.0',file=Eqns)
                print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeL'+str(k+1)+', '+str(dof+1)+', 1.0',file=Eqns)
                for m in range(4):
                    print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(B_RVE*N_GloDeriv[m][0]),file=Eqns)

        # Calling sets and setting up the MPCs for top and bottom faces
        for k in range(len(PairingFacesBT)):
            print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeB'+str(k+1)+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
            print(str(PairingFacesBT[k][0]+1),file=Sets)
            print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeT'+str(k+1)+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
            print(str(PairingFacesBT[k][1]+1),file=Sets)
            
            for dof in [0, 1, 10]:
                print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-BT'+str(k+1)+'-DOF'+str(dof+1),file=Eqns)
                print('*Equation',file=Eqns)
                print('6',file=Eqns)
                print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeT'+str(k+1)+', '+str(dof+1)+', -1.0',file=Eqns)
                print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-FaceNodeB'+str(k+1)+', '+str(dof+1)+', 1.0',file=Eqns)
                for m in range(4):
                    print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(H_RVE*N_GloDeriv[m][1]),file=Eqns)
        
                    
        # Calling sets and setting up the MPCs for vertices
        print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-V1, instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        print(str(V1+1),file=Sets)
        print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-V2, instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        print(str(V2+1),file=Sets)
        print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-V3, instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        print(str(V3+1),file=Sets)
        print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-V4, instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        print(str(V4+1),file=Sets)
        
        for dof in [0, 1]:
            print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-V23-DOF'+str(dof+1),file=Eqns)
            print('*Equation',file=Eqns)
            print('6',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V3, '+str(dof+1)+', -1.0',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V2, '+str(dof+1)+', 1.0',file=Eqns)
            for m in range(4):
                print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(H_RVE*N_GloDeriv[m][1]),file=Eqns)
                
        for dof in [0, 1, 10]:
            print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-V14-DOF'+str(dof+1),file=Eqns)
            print('*Equation',file=Eqns)
            print('6',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V4, '+str(dof+1)+', -1.0',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V1, '+str(dof+1)+', 1.0',file=Eqns)
            for m in range(4):
                print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(H_RVE*N_GloDeriv[m][1]),file=Eqns)
                
        for dof in [0, 1, 10]:
            print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-V12-DOF'+str(dof+1),file=Eqns)
            print('*Equation',file=Eqns)
            print('6',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V2, '+str(dof+1)+', -1.0',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V1, '+str(dof+1)+', 1.0',file=Eqns)
            for m in range(4):
                print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(B_RVE*N_GloDeriv[m][0]),file=Eqns)
                
        for dof in [0, 1]:
            print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-V1-DOF'+str(dof+1),file=Eqns)
            print('*Equation',file=Eqns)
            print('5',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V1, '+str(dof+1)+', -1.0',file=Eqns)
            for m in range(4):
                print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(Shape_fn[m]-0.5*B_RVE*N_GloDeriv[m][0]-0.5*H_RVE*N_GloDeriv[m][1]),file=Eqns)

        for dof in [10]:
            print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-V13-DOF'+str(dof+1),file=Eqns)
            print('*Equation',file=Eqns)
            print('6',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V3, '+str(dof+1)+', -1.0',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-V1, '+str(dof+1)+', 1.0',file=Eqns)
            for m in range(4):
                print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(B_RVE*N_GloDeriv[m][0]+H_RVE*N_GloDeriv[m][1]),file=Eqns)
        
        print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-Center'+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        print(str(center_node+1),file=Sets)
        for dof in [10]:
            print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'Center'+str(dof+1),file=Eqns)
            print('*Equation',file=Eqns)
            print('5',file=Eqns)
            print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-Center, '+str(dof+1)+', -1.0',file=Eqns)
            for m in range(4):
                print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(Shape_fn[m]),file=Eqns)

        # print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-LEFTEDGE'+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        # print(str(edgeL+1),file=Sets)
        # print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-RIGHTEDGE'+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        # print(str(edgeR+1),file=Sets)
        # print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-TOPEDGE'+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        # print(str(edgeT+1),file=Sets)
        # print('*Nset, nset=Ele'+str(i+1)+'-RVE'+str(j+1)+'-BOTTOMEDGE'+', instance=Ele'+str(i+1)+'-RVE'+str(j+1),file=Sets)
        # print(str(edgeB+1),file=Sets)

        # for dof in [10]:
        #     print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-LR-DOF'+str(dof+1),file=Eqns)
        #     print('*Equation',file=Eqns)
        #     print('6',file=Eqns)
        #     print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-LEFTEDGE, '+str(dof+1)+', -1.0',file=Eqns)
        #     print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-RIGHTEDGE, '+str(dof+1)+', 1.0',file=Eqns)
        #     for m in range(4):
        #         print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(B_RVE*N_GloDeriv[m][0]),file=Eqns)
        
        # for dof in [10]:
        #     print('** Constraint: Ele'+str(i+1)+'-RVE'+str(j+1)+'-BT-DOF'+str(dof+1),file=Eqns)
        #     print('*Equation',file=Eqns)
        #     print('6',file=Eqns)
        #     print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-TOPEDGE, '+str(dof+1)+', -1.0',file=Eqns)
        #     print('Ele'+str(i+1)+'-RVE'+str(j+1)+'-BOTTOMEDGE, '+str(dof+1)+', 1.0',file=Eqns)
        #     for m in range(4):
        #         print('Ele'+str(i+1)+'-N'+str(m+1)+', '+str(dof+1)+', '+str(H_RVE*N_GloDeriv[m][1]),file=Eqns)
Sets.close()
Eqns.close()


### Writing the new Direct FE2 input file
RVEParts = open('RVEParts.dat','r')
Insts = open('Insts.dat','r')
Sets = open('Sets.dat','r')
Eqns = open('Eqns.dat','r')
RVEMats = open('RVEMats.dat','r')

RVEParts_lines = []
Insts_lines = []
Sets_lines = []
Eqns_lines = []
RVEMats_lines = []

while 1:
    line = RVEParts.readline()
    if not line:
        break
    line = line.strip()
    RVEParts_lines.append(line)

while 1:
    line = Insts.readline()
    if not line:
        break
    line = line.strip()
    Insts_lines.append(line)
    
while 1:
    line = Sets.readline()
    if not line:
        break
    line = line.strip()
    Sets_lines.append(line)
    
while 1:
    line = Eqns.readline()
    if not line:
        break
    line = line.strip()
    Eqns_lines.append(line)
    
while 1:
    line = RVEMats.readline()
    if not line:
        break
    line = line.strip()
    RVEMats_lines.append(line)

RVEParts.close()
Insts.close()
Sets.close()
Eqns.close()
RVEMats.close()
        
f3 = open(NewInpName,'w')

Mark1 = inp1.index('*End Part')
for i in range(0,Mark1+1):
    print(inp1[i],file=f3)

for i in range(len(RVEParts_lines)):
    print(RVEParts_lines[i],file=f3)
    
Mark2 = inp1.index('*End Instance')
for i in range(Mark1+1,Mark2+2):
    print(inp1[i],file=f3)
    
for i in range(len(Insts_lines)):
    print(Insts_lines[i],file=f3)

if StartConst == 'a':
    StartConst = inp1.index('*End Assembly')    
for i in range(Mark2+2,StartConst):
    print(inp1[i],file=f3)

for i in range(len(Sets_lines)):
    print(Sets_lines[i],file=f3)
    
for i in range(len(Eqns_lines)):
    print(Eqns_lines[i],file=f3)                 
        
Mark3 = inp1.index('** ----------------------------------------------------------------')
for i in range(StartConst,Mark3):
    print(inp1[i],file=f3)

for i in range(len(RVEMats_lines)):
    print(RVEMats_lines[i],file=f3)
    
for i in range(Mark3,len(inp1)):
    print(inp1[i],file=f3)
    
f3.close()

del RVEParts_lines
del Insts_lines
del Sets_lines
del Eqns_lines
del RVEMats_lines 
os.remove('RVEParts.dat')
os.remove('Insts.dat')
os.remove('Sets.dat')
os.remove('Eqns.dat')
os.remove('RVEMats.dat')



     
        
            
        
            
            


            





























