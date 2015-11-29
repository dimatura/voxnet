"""
A dead simple and incredibly inefficient voxel renderer.
Inspired by isomer.js https://github.com/jdan/isomer.
Requires gizeh https://github.com/Zulko/gizeh.
"""

import numpy as np
import gizeh


def iso_face_depth(pts):
    return np.mean(pts[0]+pts[1]-2*pts[2])


def face_normal(face):
    a = face[:,1]-face[:,0]
    b = face[:,2]-face[:,1]
    n = np.cross(a, b)
    return n/np.linalg.norm(n)


def make_prism(origin, dxyz):
    origin = np.asarray(origin)
    dx, dy, dz = dxyz
    face1 = ((origin,
             (origin[0]+dx, origin[1]   , origin[2]   ),
             (origin[0]+dx, origin[1]   , origin[2]+dz),
             (origin[0]   , origin[1]   , origin[2]+dz)))
    face1b = [(x, y+dy, z) for (x, y, z) in face1[::-1]]
    face2 = ((origin,
             (origin[0]   , origin[1]   , origin[2]+dz),
             (origin[0]   , origin[1]+dy, origin[2]+dz),
             (origin[0]   , origin[1]+dy, origin[2]   )))
    face2b = [(x+dx, y, z) for (x, y, z) in face2[::-1]]
    face3 = ((origin,
             (origin[0]+dx, origin[1]   , origin[2]),
             (origin[0]+dx, origin[1]+dy, origin[2]),
             (origin[0]   , origin[1]+dy, origin[2])))
    face3b = [(x, y, z+dz) for (x, y, z) in face3]
    faces = [face1, face1b, face2, face2b, face3[::-1], face3b]
    return [np.asarray(face).T for face in faces]


class IsoVox(object):
    def __init__(self, width=300, height=300, scale=6.):
        self.width = width
        self.height = height
        self.scale = scale
        self.origin_x = int(.5*self.width)
        self.origin_y = int(1.1*self.height)
        self.light_angle = np.array((2., -1., 3.))
        self.light_angle /= np.linalg.norm(self.light_angle)
        self.base_color = np.array((.1, .8, .1))
        self.yaw = np.pi/6.
        self.stroke_width = 0
        self.bg_color = (1., 1., 1.)
        self.stroke_color = (.1, .1, .1)

    def _calc_M(self):
        M = np.zeros((2, 4))
        M[0, 0] = self.scale*np.cos(self.yaw)
        M[0, 1] = -self.scale*np.cos(self.yaw)
        M[0, 3] = self.origin_x
        M[1, 0] = -self.scale*np.sin(self.yaw)
        M[1, 1] = -self.scale*np.sin(self.yaw)
        M[1, 2] = -self.scale
        M[1, 3] = self.origin_y
        self.M = M

    def render(self, volume, as_html=False):
        self._calc_M()
        surf = gizeh.Surface(width=self.width, height=self.height,
                             bg_color=self.bg_color)
        ijks = np.asarray(np.nonzero(volume), dtype=np.float)
        cube = make_prism((0., 0., 0.), (1., 1., 1.))
        faces, normals = [], []
        for ijk in ijks.T:
            faces.extend([(face.T+ijk).T for face in cube])
        faces = sorted(faces, key=iso_face_depth, reverse=True)
        normals = map(face_normal, faces)
        faces_flat = np.concatenate(faces, 1)

        faces_flat_h = np.vstack((faces_flat, np.ones(faces_flat.shape[1])))
        uvs_flat = np.dot(self.M, faces_flat_h).T
        colors = np.outer(np.dot(normals, self.light_angle), self.base_color)

        for face_ix in range(0, faces_flat.shape[1], 4):
            uvs = uvs_flat[face_ix:face_ix+4]
            color = colors[face_ix//4]
            gizeh.polyline(uvs, stroke=self.stroke_color,
                    stroke_width=self.stroke_width,
                    close_path=True, fill=color).draw(surf)
        if as_html:
            return surf.get_html_embed_code()
        return surf.get_npimage()
