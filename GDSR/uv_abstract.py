import numpy as np
import os

def readTextureOBJFile(fname):
    if not(os.path.exists(fname)):
        return None, None, None, None
    VerFaceArray = []
    VerArray = []
    txtfaceArray = []
    txtArray = []
    file = open(fname, "r")
    for line in file:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            VerArray.append(v)
        if values[0] == 'vt':
            vt = [float(x) for x in values[1:3]]
            txtArray.append(vt)
        if values[0] == 'f':
            f = [int(x.split('/')[1]) for x in values[1:4]]
            txtfaceArray.append(f)
            f = [int(x.split('/')[0]) for x in values[1:4]]
            VerFaceArray.append(f)
    
    txtfaceArray = np.array(txtfaceArray)
    txtArray = np.array(txtArray)
    VerFaceArray = np.array(VerFaceArray)
    VerArray = np.array(VerArray)

    return txtArray, txtfaceArray, VerArray, VerFaceArray
    


def savePly(pDir, verts, colors, faces):
    numVerts = verts.shape[0]
    numFace = faces.shape[0]
    with open(pDir, 'w') as f:
        f.write("ply\n" + "format ascii 1.0\n")
        f.write("element vertex " + str(numVerts) + "\n")
        f.write("property float x\n" + "property float y\n" + "property float z\n")
        f.write("property uchar red\n" + "property uchar green\n"
                + "property uchar blue\n" + "property uchar alpha\n")
        f.write("element face " + str(numFace) + "\n")
        f.write("property list uchar int vertex_indices\n" + "end_header\n")
        for p in range(numVerts):
            v = verts[p]
            c = colors[p]
            f.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + " "
                    + str(int(c[0])) + " " + str(int(c[1])) + " " + str(int(c[2])) + " " + "255\n")
        for p in range(numFace):
            fds = faces[p]
            f.write("3 " + str(fds[0]-1) + " " + str(fds[1]-1)
                    + " " + str(fds[2]-1) + "\n")
        f.close()
        
    
def textureInfoGrab(fileName, saveName):
    txtArray, txtfaceArray, VerArray, VerFaceArray = readTextureOBJFile(fileName)
    print(txtArray.shape, txtfaceArray.shape)
    print(VerArray.shape, VerFaceArray.shape)
    z_t = np.zeros((txtArray.shape[0], 1))
    z_texts = np.concatenate([txtArray, z_t], axis=-1)
    
    savePly(saveName + '_uv.ply', z_texts, 192 * np.ones_like(z_texts), txtfaceArray)  # face v/vt/vn
    savePly(saveName + '_geo.ply', VerArray, 192 * np.ones_like(VerArray), VerFaceArray)
    
    

    
if __name__ == '__main__':
    caseRoot = './Data/shortskirt/Canonical/'
    caseName = 'weld/PD30'
    textureInfoGrab(caseRoot+caseName+'_C.obj', caseRoot+caseName)