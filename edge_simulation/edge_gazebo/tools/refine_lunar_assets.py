"""Build continuous textured lunar materials, retaining valid source normals."""
from pathlib import Path
import xml.etree.ElementTree as ET
from lunar_relief import deform

ROOT = Path(__file__).resolve().parents[1]
NS = 'http://www.collada.org/2005/11/COLLADASchema'
ET.register_namespace('', NS)
def tag(name):
    return f'{{{NS}}}{name}'

def textured(tree, texture, planar=False):
    root = tree.getroot()
    for name in ('library_effects', 'library_materials', 'library_images'):
        for old in root.findall(tag(name)):
            root.remove(old)
    images = ET.SubElement(root, tag('library_images'))
    ET.SubElement(ET.SubElement(images, tag('image'), id='regolith-image'), tag('init_from')).text = texture
    effect = ET.SubElement(ET.SubElement(root, tag('library_effects')), tag('effect'), id='regolith-effect')
    profile = ET.SubElement(effect, tag('profile_COMMON'))
    surface = ET.SubElement(ET.SubElement(profile, tag('newparam'), sid='regolith-surface'), tag('surface'), type='2D')
    ET.SubElement(surface, tag('init_from')).text = 'regolith-image'
    sampler = ET.SubElement(ET.SubElement(profile, tag('newparam'), sid='regolith-sampler'), tag('sampler2D'))
    ET.SubElement(sampler, tag('source')).text = 'regolith-surface'
    for axis in ('wrap_s', 'wrap_t'):
        ET.SubElement(sampler, tag(axis)).text = 'WRAP'
    lambert = ET.SubElement(ET.SubElement(profile, tag('technique'), sid='common'), tag('lambert'))
    ET.SubElement(ET.SubElement(lambert, tag('diffuse')), tag('texture'), texture='regolith-sampler', texcoord='UVMap')
    mat = ET.SubElement(ET.SubElement(root, tag('library_materials')), tag('material'), id='regolith-material')
    ET.SubElement(mat, tag('instance_effect'), url='#regolith-effect')
    for mesh in root.iter(tag('mesh')):
        if planar:
            position_id = mesh.find(tag('vertices')).find(tag('input')).get('source')[1:]
            source = next(x for x in mesh.findall(tag('source')) if x.get('id') == position_id)
            values = list(map(float, source.find(tag('float_array')).text.split()))
            uv = [v/5.0 for i,v in enumerate(values) if i%3 != 2]
            uv_source = ET.SubElement(mesh, tag('source'), id='lunar-uv')
            ET.SubElement(uv_source, tag('float_array'), id='lunar-uv-array', count=str(len(uv))).text = ' '.join(f'{v:.6f}' for v in uv)
            accessor = ET.SubElement(ET.SubElement(uv_source, tag('technique_common')), tag('accessor'), source='#lunar-uv-array', count=str(len(uv)//2), stride='2')
            for axis in ('S','T'):
                ET.SubElement(accessor, tag('param'), name=axis, type='float')
        for tri in mesh.findall(tag('triangles')):
            tri.set('material', 'regolith')
            if planar:
                inputs = tri.findall(tag('input'))
                stride = 1 + max(int(x.get('offset','0')) for x in inputs)
                offset = int(next(x for x in inputs if x.get('semantic') == 'VERTEX').get('offset','0'))
                indices = list(map(int, tri.find(tag('p')).text.split()))
                expanded=[]
                for start in range(0,len(indices),stride):
                    expanded.extend(indices[start:start+stride])
                    expanded.append(indices[start+offset])
                ET.SubElement(tri,tag('input'),semantic='TEXCOORD',source='#lunar-uv',offset=str(stride),set='0')
                tri.find(tag('p')).text=' '.join(map(str,expanded))
    for instance in root.iter(tag('instance_geometry')):
        for old in instance.findall(tag('bind_material')):
            instance.remove(old)
        common=ET.SubElement(ET.SubElement(instance,tag('bind_material')),tag('technique_common'))
        binding=ET.SubElement(common,tag('instance_material'),symbol='regolith',target='#regolith-material')
        ET.SubElement(binding,tag('bind_vertex_input'),semantic='UVMap',input_semantic='TEXCOORD',input_set='0')
    return tree

if __name__ == '__main__':
    collision = deform(ET.parse(ROOT/'models/props/mars_rough_ground/meshes/terrain.dae'))
    collision.write(ROOT/'models/props/lunar_ground/meshes/terrain.dae', encoding='utf-8', xml_declaration=True)
    for name in ('ground','hill'):
        tree=deform(ET.parse(ROOT/f'models/props/mars_rough_ground/meshes/{name}.dae'))
        textured(tree,'../materials/textures/regolith_reference.png',True).write(ROOT/f'models/props/lunar_ground/meshes/{name}.dae',encoding='utf-8',xml_declaration=True)
    for name,scale in [('lunar_rock',3.3),('lunar_rock_small',1.575)]:
        folder=ROOT/'models/props'/name
        tree=ET.parse(ROOT/'models/props/falling_rock3/meshes/FallingRock03.dae')
        textured(tree,'../../lunar_ground/materials/textures/regolith_reference.png').write(folder/'meshes/rock.dae',encoding='utf-8',xml_declaration=True)
        sdf=ET.parse(folder/'model.sdf')
        for mesh in sdf.iter('mesh'):
            for old in mesh.findall('scale'):
                mesh.remove(old)
            ET.SubElement(mesh,'scale').text=f'{scale} {scale} {scale}'
        # Dark grey obstacles stand out against the lighter lunar ground.
        for visual in sdf.iter('visual'):
            material = visual.find('material')
            if material is None:
                material = ET.SubElement(visual, 'material')
            for key, value in (('ambient', '0.80 0.80 0.80 1'),
                               ('diffuse', '1.0 1.0 1.0 1'),
                               ('specular', '0 0 0 1')):
                element = material.find(key)
                if element is None:
                    element = ET.SubElement(material, key)
                element.text = value
            # Restore rock detail with the source albedo, normal and roughness maps.
            old_pbr = material.find('pbr')
            if old_pbr is not None:
                material.remove(old_pbr)
            metal = ET.SubElement(ET.SubElement(material, 'pbr'), 'metal')
            texture_root = 'model://props/falling_rock3/materials/textures/'
            for key, value in (
                ('albedo_map', texture_root + 'FallingRocks_Albedo.jpg'),
                ('normal_map', texture_root + 'FallingRocks_Normal.jpg'),
                ('roughness_map', texture_root + 'FallingRocks_Roughness.jpg'),
                ('roughness', '1.0'), ('metalness', '0.0')):
                ET.SubElement(metal, key).text = value
        sdf.write(folder/'model.sdf',encoding='utf-8',xml_declaration=True)
