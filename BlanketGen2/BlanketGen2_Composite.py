import bpy
import os
import json
import sys



with open("config_composite.json", "r") as f:
    config = json.load(f)
    path_to_dataset = config["path_to_dataset"]
    path_to_composited = config["path_to_save_composited"]
    path_to_render = config["path_to_save_render"]

try:
    os.mkdir(path_to_composited)
except OSError as error:
    print(error)  

bpy.ops.wm.read_factory_settings()
[bpy.data.objects.remove(obj) for obj in bpy.data.objects]
[bpy.data.materials.remove(m) for m in bpy.data.materials]

actions = []
for (dirpath, dirnames, filenames) in os.walk(path_to_render):
    if "webp" not in str(filenames):
        continue
    if dirpath.split("/")[-3] not in actions:
        actions.append(dirpath.split("/")[-3])
    continue

for base in actions:
    person = "s04"
    camera = "60457274"

    path_to_blanket_folders = path_to_render+"/"+base+"/"
    
    blanket_paths = {}
    for (dirpath, dirnames, filenames) in os.walk(path_to_blanket_folders):
        for i in dirnames:
            blanket_paths[i] = dirpath+"/"+i
        break

    bpy.context.scene.render.resolution_x = 900
    bpy.context.scene.render.resolution_y = 900
    bpy.context.scene.view_settings.view_transform = "Standard"
    bpy.context.scene.render.image_settings.file_format = "FFMPEG"
    bpy.context.scene.render.ffmpeg.format = "MPEG4"
    bpy.context.scene.render.filepath = path_to_composited
    bpy.context.scene.render.fps=50

    blanket_lengths = {}
    blanket_starts = {}

    for blanket_num in (blanket_paths):
        with open(blanket_paths[blanket_num]+"/"+camera+"/vid_info.json") as f:
            vid_info=json.load(f)
        blanket_starts[blanket_num] = vid_info["frame_start"]
        blanket_lengths[blanket_num] = vid_info["frame_end"]-vid_info["frame_start"]+1
    
        
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree   
    [tree.nodes.remove(node) for node in tree.nodes] 

    # Create output node
    comp_node = tree.nodes.new('CompositorNodeComposite') 

    # Create alpha over node
    alpha_node=tree.nodes.new('CompositorNodeAlphaOver') 
    alpha_node.use_premultiply = True

    # Create the node for the original video
    og_node = tree.nodes.new('CompositorNodeImage') 
    path_to_og = path_to_dataset+"/train/"+person+"/videos/"+camera+"/"
    bpy.ops.image.open(filepath=path_to_og+base+".mp4", directory=path_to_og, files=[{"name":base+".mp4"}])
    bpy.data.images[base+".mp4"].source = "MOVIE"
    og_node.image = bpy.data.images[base+".mp4"]
    og_node.frame_duration = 100000
    og_node.frame_start = 0
    og_node.frame_offset = 0

    comp_node=tree.nodes["Composite"]

    # Link nodes
    links = tree.links
    links.new(og_node.outputs[0], alpha_node.inputs[1])
    links.new(alpha_node.outputs[0], comp_node.inputs[0])

    # Create input images nodes
    for blanket_num in blanket_paths:
        if blanket_lengths[blanket_num] < 2:
            continue
        # print(blanket_starts[blanket_num],blanket_lengths[blanket_num])
        blanket_path = f"{path_to_blanket_folders}/{blanket_num}/{camera}/"
        print(blanket_path, file=sys.stdout, flush=True)
        bpy.ops.image.open(filepath=f"{blanket_path}/image_{blanket_starts[blanket_num]:05d}.webp",
        directory=blanket_path,
        files=[{"name":f"/image_{frame:05d}.webp"} for frame in range(blanket_starts[blanket_num],blanket_starts[blanket_num]+blanket_lengths[blanket_num]-1)])
        blanket_node = tree.nodes.new(type='CompositorNodeImage')
        bpy.data.images[f"image_{blanket_starts[blanket_num]:05d}.webp"].name = f"{blanket_num}"
        blanket_node.image = bpy.data.images[f"{blanket_num}"]
        blanket_node.frame_duration = blanket_lengths[blanket_num]
        blanket_node.frame_start = 0
        blanket_node.frame_offset = -1+blanket_starts[blanket_num]
        og_node.frame_offset = blanket_starts[blanket_num]

        links.new(blanket_node.outputs[0], alpha_node.inputs[2])

        bpy.data.scenes["Scene"].frame_start = 0
        bpy.data.scenes["Scene"].frame_end = blanket_lengths[blanket_num]-2
        bpy.ops.render.render(animation=True)

        for (dirpath, dirnames, filenames) in os.walk(path_to_composited):
            for i in filenames:
                if "-" in i:
                    os.rename(dirpath+"/"+i, f"{dirpath}/{base}_{blanket_num}.mp4")
