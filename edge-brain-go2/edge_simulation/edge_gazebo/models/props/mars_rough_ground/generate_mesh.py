"""Generate a 100 m rough surface with a level spawn area (metres)."""
from pathlib import Path
import math
import copy
import xml.etree.ElementTree as ET


def height(x, y):
    fade = min(1.0, max(0.0, (math.hypot(x, y) - 1.5) / 2.0))
    fade = fade * fade * (3.0 - 2.0 * fade)
    # A smooth 0.4 m hill centred 8 m ahead, tapering to zero at 4 m.
    radius = math.hypot(x - 8.0, y)
    hill = 0.4 * (1.0 - (radius / 4.0)**2)**2 if radius < 4.0 else 0.0
    return hill + fade * (0.07 * math.sin(2.1*x + 0.7*y)
                   + 0.045 * math.sin(1.3*y - 0.9*x)
                   + 0.025 * math.sin(4.0*x + 2.7*y))


def generate():
    n, step = 201, 0.5
    vertices, normals, faces = [], [], []
    for j in range(n):
        for i in range(n):
            x, y = -50 + i*step, -50 + j*step
            vertices.extend((x, y, height(x, y)))
            eps = 0.01
            dx = (height(x+eps,y)-height(x-eps,y))/(2*eps)
            dy = (height(x,y+eps)-height(x,y-eps))/(2*eps)
            length = math.sqrt(dx*dx+dy*dy+1)
            normals.extend((-dx/length, -dy/length, 1/length))
    for j in range(n-1):
        for i in range(n-1):
            a = j*n+i
            for index in (a, a+1, a+n, a+1, a+n+1, a+n):
                faces.extend((index,index))
    ET.register_namespace('', 'http://www.collada.org/2005/11/COLLADASchema')
    root = ET.Element('COLLADA', xmlns='http://www.collada.org/2005/11/COLLADASchema', version='1.4.1')
    asset = ET.SubElement(root,'asset')
    ET.SubElement(asset,'unit',name='meter',meter='1')
    ET.SubElement(asset,'up_axis').text='Z_UP'
    mesh=ET.SubElement(ET.SubElement(ET.SubElement(root,'library_geometries'),'geometry',id='terrain'),'mesh')
    for name, values in [('positions',vertices),('normals',normals)]:
        source=ET.SubElement(mesh,'source',id=name)
        ET.SubElement(source,'float_array',id=name+'-array',count=str(len(values))).text=' '.join(f'{v:.6f}' for v in values)
        accessor=ET.SubElement(ET.SubElement(source,'technique_common'),'accessor',source='#'+name+'-array',count=str(n*n),stride='3')
        for axis in 'XYZ': ET.SubElement(accessor,'param',name=axis,type='float')
    v=ET.SubElement(mesh,'vertices',id='vertices')
    ET.SubElement(v,'input',semantic='POSITION',source='#positions')
    tri=ET.SubElement(mesh,'triangles',count=str(len(faces)//6))
    ET.SubElement(tri,'input',semantic='VERTEX',source='#vertices',offset='0')
    ET.SubElement(tri,'input',semantic='NORMAL',source='#normals',offset='1')
    ET.SubElement(tri,'p').text=' '.join(map(str,faces))
    scene=ET.SubElement(ET.SubElement(root,'library_visual_scenes'),'visual_scene',id='Scene')
    ET.SubElement(ET.SubElement(scene,'node',id='Terrain'),'instance_geometry',url='#terrain')
    ET.SubElement(ET.SubElement(root,'scene'),'instance_visual_scene',url='#Scene')
    ET.ElementTree(root).write(Path(__file__).parent/'meshes/terrain.dae',encoding='utf-8',xml_declaration=True)
    # Split only the visible surface; keep the complete collision mesh above.
    groups = {'ground': [], 'hill': []}
    for offset in range(0, len(faces), 6):
        triangle = faces[offset:offset+6]
        ids = triangle[::2]
        cx = sum(vertices[k*3] for k in ids)/3
        cy = sum(vertices[k*3+1] for k in ids)/3
        group = 'hill' if math.hypot(cx-8.0, cy) < 4.0 else 'ground'
        groups[group].extend(triangle)
    for name, indices in groups.items():
        visual = copy.deepcopy(root)
        triangles = visual.find('.//triangles')
        triangles.set('count', str(len(indices)//6))
        triangles.find('p').text = ' '.join(map(str, indices))
        ET.ElementTree(visual).write(Path(__file__).parent/'meshes'/f'{name}.dae', encoding='utf-8', xml_declaration=True)



if __name__ == '__main__':
    generate()
