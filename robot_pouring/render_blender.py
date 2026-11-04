"""Render saved fluid surfaces and actual MuJoCo robot poses in Blender."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import bpy
from mathutils import Matrix, Vector

ROOT = Path(__file__).resolve().parents[1]
for key, folder in [("XDG_CACHE_HOME", "blender"), ("CUDA_CACHE_PATH", "cuda"), ("OPTIX_CACHE_PATH", "optix")]:
    os.environ.setdefault(key, str(ROOT / ".cache" / folder))


def material(name, color, roughness=.3, metal=0., transmission=0.):
    m = bpy.data.materials.new(name); m.diffuse_color = (*color[:3], 1); m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color[:3], 1)
    b.inputs["Roughness"].default_value = roughness
    b.inputs["Metallic"].default_value = metal
    b.inputs["Transmission Weight"].default_value = transmission
    b.inputs["IOR"].default_value = 1.333 if name == "Water" else 1.46
    return m


def mesh_object(name, vertices, faces, mat):
    mesh = bpy.data.meshes.new(name); mesh.from_pydata(vertices.tolist() if isinstance(vertices, np.ndarray) else vertices, [], faces.tolist() if isinstance(faces, np.ndarray) else faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh); bpy.context.collection.objects.link(obj)
    if mat: mesh.materials.append(mat)
    for face in mesh.polygons: face.use_smooth = True
    return obj


def cylinder(name, r, height, pos, mat, parent=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=r, depth=height, location=pos)
    obj = bpy.context.object; obj.name = name; obj.data.materials.append(mat)
    if parent: obj.parent = parent
    for p in obj.data.polygons: p.use_smooth = True
    bevel = obj.modifiers.new("Soft edge", "BEVEL"); bevel.width = .0015; bevel.segments = 3
    return obj


def vessel(name, r, h, wall, mat, parent):
    # Separate normal seams at horizontal surfaces; one real vertex per disk
    # center avoids degenerate center quads and bevel-generated radial artifacts.
    vertices, faces, smooth = [], [], []
    bands = [(r+wall,-wall,r+wall,h,True),(r+wall,h,r,h,False),(r,h,r,0,True)]
    for r0,z0,r1,z1,shade in bands:
        base=len(vertices)
        for radius,z in [(r0,z0),(r1,z1)]:
            vertices += [(radius*math.cos(2*math.pi*i/96),radius*math.sin(2*math.pi*i/96),z) for i in range(96)]
        faces += [(base+i,base+(i+1)%96,base+96+(i+1)%96,base+96+i) for i in range(96)]
        smooth += [shade]*96
    for radius,z,up in [(r,0,True),(r+wall,-wall,False)]:
        base=len(vertices);vertices += [(radius*math.cos(2*math.pi*i/96),radius*math.sin(2*math.pi*i/96),z) for i in range(96)]
        center=len(vertices);vertices.append((0,0,z))
        faces += [(center,base+i,base+(i+1)%96) if up else (center,base+(i+1)%96,base+i) for i in range(96)]
        smooth += [False]*96
    obj = mesh_object(name, vertices, faces, mat); obj.parent = parent
    for polygon,shade in zip(obj.data.polygons,smooth):polygon.use_smooth=shade
    return obj


def curve(name, points, radius, mat, parent):
    c = bpy.data.curves.new(name, "CURVE"); c.dimensions = "3D"; c.bevel_depth = radius; c.bevel_resolution = 4
    s = c.splines.new("POLY"); s.points.add(len(points)-1)
    for p, xyz in zip(s.points, points): p.co = (*xyz, 1)
    o = bpy.data.objects.new(name, c); bpy.context.collection.objects.link(o); o.data.materials.append(mat); o.parent = parent
    return o


def empty(name):
    obj = bpy.data.objects.new(name, None); bpy.context.collection.objects.link(obj); return obj


def area(name, pos, target, energy, size, color):
    light = bpy.data.lights.new(name, "AREA"); light.energy = energy; light.shape = "DISK"; light.size = size; light.color = color
    obj = bpy.data.objects.new(name, light); bpy.context.collection.objects.link(obj); obj.location = pos
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat("-Z", "Y").to_euler()


def setup(meta, args):
    bpy.ops.object.select_all(action="SELECT"); bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene; scene.render.engine = "CYCLES"; scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True; scene.cycles.device = "GPU"; scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 10; scene.render.use_persistent_data = True; scene.cycles.seed = 19
    prefs = bpy.context.preferences.addons["cycles"].preferences; prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices: d.use = d.type == "OPTIX"
    scene.render.resolution_x = args.width; scene.render.resolution_y = args.height; scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"; scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -.8
    scene.world.use_nodes = True; scene.world.node_tree.nodes["Background"].inputs[0].default_value = (.13,.17,.22,1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = .35
    floor = material("Matte slate", (.014,.022,.035), .6)
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0,0,-.065)); bpy.context.object.data.materials.append(floor)
    tabletop = material("Laboratory worktop", (.035,.055,.075), .33, .3)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(.38,0,-.032)); table = bpy.context.object; table.name = "Worktop"
    table.scale = (1.25,.90,.06); bpy.ops.object.transform_apply(location=False, rotation=False, scale=True); table.data.materials.append(tabletop)
    bevel = table.modifiers.new("Rounded worktop", "BEVEL"); bevel.width = .025; bevel.segments = 5
    gold = material("Brass details", (.42,.23,.08), .24, .75)
    dark = material("Graphite", (.025,.04,.055), .3, .5)
    cylinder("Robot base adapter", .105, .012, (0,0,-.005), dark)
    cfg = meta["config"]
    target, cup = empty("Target vessel"), empty("Robot held pitcher")
    cupmat = material("Clear blue borosilicate", (.86,.96,.98), .07, transmission=1.)
    vessel("Open pitcher", cfg["cup_radius"], cfg["cup_height"], cfg["cup_wall"], cupmat, cup)
    points = [(-.047,0,.105),(-.083,0,.105),(-.101,0,.092),(-.106,0,.05),(-.086,0,.027),(-.047,0,.027)]
    curve("Pitcher handle", points, .006, gold, cup)
    rim = material("Ivory ceramic", (.72,.84,.84), .42)
    if cfg["task"] == "flower": rim = material("Terracotta", (.48,.17,.075), .55)
    vessel("Receiving vessel", cfg["target_radius"], cfg["target_height"], .008, rim, target)
    soil = None
    if cfg["task"] == "flower":
        soilmat = material("Soil wetting", (.18,.07,.025), .95)
        nodes = soilmat.node_tree.nodes; links = soilmat.node_tree.links
        att = nodes.new("ShaderNodeAttribute"); att.attribute_name = "wetness"
        mix = nodes.new("ShaderNodeMixRGB"); mix.inputs[1].default_value = (.16,.065,.025,1); mix.inputs[2].default_value = (.025,.014,.009,1)
        links.new(att.outputs["Fac"], mix.inputs[0]); links.new(mix.outputs[0], nodes.get("Principled BSDF").inputs["Base Color"])
        vertices = []
        for j in range(17):
            radius = cfg["target_radius"] * j / 16
            for i in range(96): vertices.append((radius*math.cos(2*math.pi*i/96),radius*math.sin(2*math.pi*i/96),.0502))
        faces = [(j*96+i,j*96+(i+1)%96,(j+1)*96+(i+1)%96,(j+1)*96+i) for j in range(16) for i in range(96)]
        soil = mesh_object("Absorbing soil surface", vertices, faces, soilmat); soil.parent = target
        soil.data.attributes.new("wetness", "FLOAT", "POINT")
        green = material("Living foliage", (.08,.29,.025), .42)
        stem = material("Plant stem", (.14,.23,.04), .55)
        cylinder("Plant stem", .007, .22, (0,0,.16), stem, target)
        for i in range(15):
            angle = i*2.399; z = .09 + i*.010
            end = (.035*math.cos(angle),.035*math.sin(angle),z+.016)
            curve("Leaf petiole", [(0,0,z),end], .0016, stem, target)
            bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=10, location=end)
            leaf=bpy.context.object; leaf.name="Leaf"; leaf.parent=target; leaf.scale=(.029,.011,.002)
            leaf.rotation_euler=(.15, -.4, angle); leaf.data.materials.append(green)
            for face in leaf.data.polygons: face.use_smooth=True
        petal = material("Coral petals", (.62,.055,.12), .38)
        for j in range(5):
            a=2*math.pi*j/5
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=(.016*math.cos(a),.016*math.sin(a),.268))
            flower=bpy.context.object; flower.parent=target; flower.scale=(.018,.011,.006); flower.rotation_euler[2]=a; flower.data.materials.append(petal)
        cylinder("Flower center", .008, .008, (0,0,.273), gold, target)
    water = material("Water", (.12,.54,.72), .07, transmission=.88)
    area("Large softbox", (.2,-.8,2.0), (.4,0,.2), 100, 1.4, (1.,.9,.78))
    area("Cool rim", (.3,.8,1.3), (.4,0,.3), 80, 1.0, (.65,.82,1.))
    area("Water highlight", (1.2,-.15,.9), (.48,0,.2), 50, .65, (1.,1.,1.))
    bpy.ops.object.camera_add(); camera=bpy.context.object; camera.data.lens=48; scene.camera=camera
    return scene,camera,cup,target,water,soil


def camera_fit(meta, args, robot):
    """One constant distance fits the whole recorded episode without zoom pumping."""
    look = np.array([.34, 0., .35])
    tanx = 36. / (2. * 48.)
    tany = tanx * args.height / args.width
    factor = 1.
    bounds = [np.asarray(o.bound_box) for o in robot]
    cfg = meta["config"]
    cup_bounds = np.array([(x,y,z) for x in [-.112,.052] for y in [-.052,.052] for z in [-.008,.142]])
    r = cfg["target_radius"] + .012
    target_bounds = np.array([(x,y,z) for x in [-r,r] for y in [-r,r] for z in [-.01,.29 if cfg["task"] == "flower" else cfg["target_height"]+.01]])
    for i in meta["indices"]:
        with np.load(args.input/f"{i:05d}.npz") as d:
            parts = [b @ d["geom_mat"][j].T + d["geom_pos"][j] for j,b in enumerate(bounds)]
            parts += [cup_bounds @ d["cup_rot"].T + d["cup_pos"], target_bounds + d["target"]]
            points = np.concatenate(parts)
        t=i/max(1,meta["frames"]-1);angle=math.radians(-58+24*t)
        offset=np.array([1.8*math.cos(angle),1.8*math.sin(angle),.85]);distance=np.linalg.norm(offset)
        forward=-offset/distance;right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward)
        delta=points-look;depth=delta@forward
        required=np.maximum(np.abs(delta@right)/(.84*tanx)-depth,np.abs(delta@up)/(.80*tany)-depth)
        factor=max(factor,float(required.max()/distance))
    return look, factor


def main():
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--indices");p.add_argument("--width",type=int,default=1280);p.add_argument("--height",type=int,default=720)
    p.add_argument("--samples",type=int,default=48);p.add_argument("--resume",action="store_true")
    args=p.parse_args(sys.argv[sys.argv.index("--")+1:]);args.output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((args.input/"manifest.json").read_text()); scene,camera,cup,target,water,soil=setup(meta,args)
    robot_data=np.load(args.input/"robot_meshes.npz");robot=[]
    for spec in meta["geoms"]:
        mat=material("Panda "+str(spec["id"]),np.power(spec["rgba"][:3],2.2),.3,.15)
        obj=mesh_object(spec["body"]+"_"+str(spec["id"]),robot_data[spec["mesh"]+"_vertices"],robot_data[spec["mesh"]+"_faces"],mat);robot.append(obj)
    indices=[int(i) for i in args.indices.split(",")] if args.indices else meta["indices"]
    look, camera_scale = camera_fit(meta,args,robot)
    settings=dict(width=args.width,height=args.height,samples=args.samples,
                  camera_scale=camera_scale,lookat=look.tolist(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input_manifest_sha256=hashlib.sha256((args.input/"manifest.json").read_bytes()).hexdigest())
    settings_path=args.output/"render_settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError("Render settings changed; use a fresh output directory")
    temporary=settings_path.with_name(f'.render-settings-{os.getpid()}.tmp')
    temporary.write_text(json.dumps(settings,indent=2));temporary.replace(settings_path)
    liquid=None
    for i in indices:
        filename=args.output/f"{i:05d}.png"
        if args.resume and filename.exists():continue
        d=np.load(args.input/f"{i:05d}.npz")
        for j,obj in enumerate(robot):
            transform=Matrix(d["geom_mat"][j].tolist()).to_4x4();transform.translation=Vector(d["geom_pos"][j]);obj.matrix_world=transform
        transform=Matrix(d["cup_rot"].tolist()).to_4x4();transform.translation=Vector(d["cup_pos"]);cup.matrix_world=transform
        target.location=d["target"]
        if liquid:
            old=liquid.data;bpy.data.objects.remove(liquid,do_unlink=True);bpy.data.meshes.remove(old)
        liquid=mesh_object("Simulated water",d["vertices"],d["faces"],water)
        if soil and len(d["absorbed"]):
            points=np.asarray([v.co[:] for v in soil.data.vertices])+d["target"]
            delta=points[:,None,:2]-d["absorbed"][None,:,:2]
            wet=1-np.exp(-.15*np.exp(-np.sum(delta*delta,axis=2)/.0003).sum(1))
            soil.data.attributes["wetness"].data.foreach_set("value",wet.astype(np.float32));soil.data.update()
        t=i/max(1,meta["frames"]-1);angle=math.radians(-58+24*t)
        camera.location=(look+camera_scale*np.array([1.8*math.cos(angle),1.8*math.sin(angle),.85])).tolist()
        camera.rotation_euler=(Vector(look)-camera.location).to_track_quat("-Z","Y").to_euler()
        scene.render.filepath=str(filename);bpy.ops.render.render(write_still=True)
    (args.output/("render_"+str(indices[0])+"_"+str(indices[-1])+".json")).write_text(json.dumps(dict(indices=indices,width=args.width,height=args.height,samples=args.samples),indent=2))


if __name__=="__main__":main()
