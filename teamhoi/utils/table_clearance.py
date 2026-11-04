"""Exact lowest point of fixed box/cylinder URDF collision geometry."""
import itertools
import math
import xml.etree.ElementTree as ET
import torch


def _origin(element):
    transform = torch.eye(4, dtype=torch.float64)
    if element is None:
        return transform
    xyz = [float(x) for x in element.get('xyz', '0 0 0').split()]
    roll, pitch, yaw = [float(x) for x in element.get('rpy', '0 0 0').split()]
    cr,sr,cp,sp,cy,sy = math.cos(roll),math.sin(roll),math.cos(pitch),math.sin(pitch),math.cos(yaw),math.sin(yaw)
    transform[:3,:3] = torch.tensor([[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                                    [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],
                                    [-sp,cp*sr,cp*cr]], dtype=torch.float64)
    transform[:3,3] = torch.tensor(xyz, dtype=torch.float64)
    return transform


def load_collision_geometry(path, device='cpu', dtype=torch.float32):
    """Parse once; retain exact box vertices and analytic cylinder parameters."""
    root = ET.parse(path).getroot()
    links = {link.get('name'):link for link in root.findall('link')}
    joints = root.findall('joint')
    children = {j.find('child').get('link') for j in joints}
    roots = set(links)-children
    if len(roots) != 1:
        raise ValueError('Expected one URDF root link')
    frames = {next(iter(roots)):torch.eye(4,dtype=torch.float64)}
    pending = list(joints)
    while pending:
        progress = False
        for joint in list(pending):
            if joint.get('type') != 'fixed':
                raise ValueError('Clearance supports fixed rigid tables only')
            parent = joint.find('parent').get('link')
            if parent not in frames:
                continue
            frames[joint.find('child').get('link')] = frames[parent] @ _origin(joint.find('origin'))
            pending.remove(joint)
            progress = True
        if not progress:
            raise ValueError('Disconnected or cyclic URDF')
    corners, centers, axes, radii, lengths = [],[],[],[],[]
    for name,link in links.items():
        for collision in link.findall('collision'):
            transform = frames[name] @ _origin(collision.find('origin'))
            geometry = collision.find('geometry')
            box, cylinder = geometry.find('box'), geometry.find('cylinder')
            if box is not None:
                half = torch.tensor([float(x)/2 for x in box.get('size').split()],dtype=torch.float64)
                vertices = torch.tensor(list(itertools.product((-1.,1.),repeat=3)),dtype=torch.float64)*half
                corners.append(vertices @ transform[:3,:3].T + transform[:3,3])
            elif cylinder is not None:
                centers.append(transform[:3,3]);axes.append(transform[:3,2])
                radii.append(float(cylinder.get('radius')))
                lengths.append(float(cylinder.get('length'))/2)
            else:
                raise ValueError('Unsupported collision geometry in '+str(path))
    if not corners and not centers:
        raise ValueError('Table has no supported collision geometry')
    convert = lambda value: value.to(device=device,dtype=dtype)
    return dict(corners=convert(torch.cat(corners) if corners else torch.empty(0,3)),
                centers=convert(torch.stack(centers) if centers else torch.empty(0,3)),
                axes=convert(torch.stack(axes) if axes else torch.empty(0,3)),
                radii=convert(torch.tensor(radii)), half_lengths=convert(torch.tensor(lengths)))


def lowest_collision_offset(rotation, geometry):
    """Lowest geometry z relative to root z, for a batch of xyzw rotations."""
    q = rotation / rotation.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    x,y,z,w = q.unbind(-1)
    row = torch.stack((2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),-1)
    lowest = rotation.new_full(rotation.shape[:-1],float('inf'))
    if geometry['corners'].shape[0]:
        lowest = (row @ geometry['corners'].T).min(-1).values
    if geometry['centers'].shape[0]:
        axis_z = (row @ geometry['axes'].T).clamp(-1.,1.)
        extent = (geometry['half_lengths']*axis_z.abs()
                  + geometry['radii']*torch.sqrt((1-axis_z.square()).clamp_min(0.)))
        cylinder_low = (row @ geometry['centers'].T - extent).min(-1).values
        lowest = torch.minimum(lowest,cylinder_low)
    return lowest


def rotation_height_limit(lowest_offset, target_clearance, margin):
    """Preserve the old 1.30m cap unless geometry requires extra headroom."""
    needed = target_clearance-lowest_offset+margin
    return torch.maximum(torch.full_like(needed,1.30),needed)
