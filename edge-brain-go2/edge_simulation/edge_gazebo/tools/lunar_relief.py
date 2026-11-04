"""Seeded lunar depressions; identical deformation for rendering and collision."""
import math
import random
import xml.etree.ElementTree as ET

NS = 'http://www.collada.org/2005/11/COLLADASchema'
def tag(name):
    return f'{{{NS}}}{name}'

rng = random.Random(20261101)
CRATERS = []
for count, bounds in [(42, (-44, 44, -44, 44)), (12, (-24, 30, -20, 20))]:
    for _ in range(count):
        x = rng.uniform(bounds[0], bounds[1])
        y = rng.uniform(bounds[2], bounds[3])
        radius = rng.uniform(3.0, 7.5)
        CRATERS.append((x, y, radius, radius*rng.uniform(0.65, 1.3), rng.uniform(0.22, 0.65), rng.uniform(0, math.tau)))

def relief(x, y):
    # Warped wide channels and weathered banks, as in the reference image.
    wx = x + 1.2*math.sin(y*0.16)
    wy = y + 0.85*math.sin(x*0.21)
    z = 0.12*math.sin(wx*0.22+wy*0.17)*math.sin(wy*0.28)
    z += 0.035*math.sin(wx*1.05+wy*0.52) + 0.018*math.sin(wx*2.1-wy*1.3)
    channel = 0.9*math.sin(x*0.14) + 0.35*math.sin(x*0.39)
    along = math.exp(-(x/37.0)**4)
    z += along*(0.72*math.exp(-((y-channel-5.3)/2.1)**2)
                + 0.48*math.exp(-((y-channel+5.8)/2.7)**2)
                - 0.16*math.exp(-((y-channel)/2.4)**2))
    z += 0.32*math.exp(-((wx+13)/9)**2-((wy-12)/5)**2)
    for cx, cy, rx, ry, depth, angle in CRATERS:
        dx, dy = x-cx, y-cy
        if abs(dx)>rx*1.8 or abs(dy)>rx*1.8:
            continue
        u = (dx*math.cos(angle)+dy*math.sin(angle))/rx
        v = (-dx*math.sin(angle)+dy*math.cos(angle))/ry
        r = math.sqrt(u*u+v*v)
        if r < 1.6:
            # Broad bowl and a low, rounded, weathered rim.
            z += -depth*math.exp(-2.8*r*r) + depth*0.32*math.exp(-((r-0.98)/0.32)**2)
    t = min(1.0, max(0.0, (math.hypot(x,y)-2.0)/2.0))
    spawn_blend = t*t*(3-2*t)
    edge_blend = min(1.0, max(0.0, (50-max(abs(x),abs(y)))/4.0))
    return z*spawn_blend*edge_blend

def deform(tree):
    for mesh in tree.getroot().iter(tag('mesh')):
        vertices = mesh.find(tag('vertices'))
        position_id = next(i for i in vertices.findall(tag('input')) if i.get('semantic')=='POSITION').get('source')[1:]
        source = next(s for s in mesh.findall(tag('source')) if s.get('id')==position_id)
        array = source.find(tag('float_array'))
        numbers = list(map(float,array.text.split()))
        points = [numbers[i:i+3] for i in range(0,len(numbers),3)]
        normals = []
        for p in points:
            x,y=p[:2]
            p[2] += relief(x,y)
            # Existing surface is a regular grid; calculate full geometric normals below.
        sums = [[0.0,0.0,0.0] for _ in points]
        for tri in mesh.findall(tag('triangles')):
            inputs=tri.findall(tag('input'))
            stride=1+max(int(i.get('offset','0')) for i in inputs)
            vo=int(next(i for i in inputs if i.get('semantic')=='VERTEX').get('offset','0'))
            normal_input=next(i for i in inputs if i.get('semantic')=='NORMAL')
            no=int(normal_input.get('offset','0'))
            indices=list(map(int,tri.find(tag('p')).text.split()))
            for start in range(0,len(indices),stride*3):
                ids=[indices[start+j*stride+vo] for j in range(3)]
                a,b,c=[points[i] for i in ids]
                u=[b[k]-a[k] for k in range(3)];v=[c[k]-a[k] for k in range(3)]
                n=[u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]
                for j,i in enumerate(ids):
                    for k in range(3): sums[i][k]+=n[k]
                    indices[start+j*stride+no]=i
            normal_input.set('source','#relief-normals')
            tri.find(tag('p')).text=' '.join(map(str,indices))
        for n in sums:
            length=math.sqrt(sum(v*v for v in n))
            normals.extend([v/length for v in n] if length else [0,0,1])
        array.text=' '.join(f'{v:.6f}' for p in points for v in p)
        src=ET.SubElement(mesh,tag('source'),id='relief-normals')
        ET.SubElement(src,tag('float_array'),id='relief-normals-array',count=str(len(normals))).text=' '.join(f'{v:.7f}' for v in normals)
        accessor=ET.SubElement(ET.SubElement(src,tag('technique_common')),tag('accessor'),source='#relief-normals-array',count=str(len(points)),stride='3')
        for axis in 'XYZ': ET.SubElement(accessor,tag('param'),name=axis,type='float')
    return tree
