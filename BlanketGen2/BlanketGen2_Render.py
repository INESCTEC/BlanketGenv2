import bpy
import json
from os import walk
import os
import time
import copy
import numpy as np

# 
# Sets the render parameters with sane defaults
# 
def renderParams(engine="CYCLES", render_params={}):
    print(render_params)
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.image_settings.file_format = 'WEBP'
    bpy.context.scene.render.image_settings.quality = 100
    bpy.context.scene.view_settings.view_transform = "Standard"

    if engine=="BLENDER_EEVEE":
        bpy.context.scene.eevee.taa_render_samples = render_params["samples_per_pixel"] if "samples_per_pixel" in render_params else 64
        bpy.context.scene.eevee.use_gtao = render_params["ambient_occlusion"] if "ambient_occlusion" in render_params else True
        bpy.context.scene.eevee.use_bloom = render_params["bloom"] if "bloom" in render_params else True
        bpy.context.scene.eevee.use_ssr = render_params["ssr"] if "ssr" in render_params else True
        bpy.context.scene.eevee.use_motion_blur = render_params["motion_blur"] if "motion_blur" in render_params else False

    if engine=="CYCLES":    
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.preferences.addons['cycles'].preferences.get_devices()
        for device in bpy.context.preferences.addons['cycles'].preferences.devices:
            device.use="nvidia" in device.id.lower()
            print(device, device.use)
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        bpy.context.scene.cycles.use_adaptive_sampling = render_params["adaptive_sampling"] if "adaptive_sampling" in render_params else True
        if bpy.context.scene.cycles.use_adaptive_sampling:
            bpy.context.scene.cycles.adaptive_threshold = render_params["threshold"] if "threshold" in render_params else 0.1
            bpy.context.scene.cycles.adaptive_min_samples = render_params["min_samples"] if "min_samples" in render_params else 0
        bpy.context.scene.cycles.samples = render_params["max_samples"] if "max_samples" in render_params else 20
        bpy.context.scene.cycles.time_limit = render_params["time_limit"] if "time_limit" in render_params else 1
        bpy.context.scene.cycles.use_denoising = render_params["use_denoising"] if "use_denoising" in render_params else True

    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.render.use_overwrite = True
    return

# 
# Creates a new material and adds it to the specified object. If no object is passed, it adds it to the currently active object in the context
# 
def changeMaterial(name="Material",color_over1=(0.9,0.9,0.9,1), color_over2=(0.1,0.1,0.1,1), color_under=(0,0,0,1), ob=None, 
                holdout=False, wave_scale=150, wave_distortion=7, noise_scale=30, noise_distortion=7, checker_scale=10,
                bump_wave=0.4, bump_noise1=0.2, bump_noise2=0.3, pattern="checker"):
    if ob is None:
        ob = bpy.context.object

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes=True

    # Assign it to object
    if ob.data.materials:
        # assign to 1st material slot
        ob.data.materials[0] = mat
    else:
        # no slots
        ob.data.materials.append(mat)
    if holdout:
        ob.is_holdout=True
    else:
        ob.is_holdout=False
        mat.cycles.displacement_method="BOTH"
        
        nodes=mat.node_tree.nodes
        
        nodes["Principled BSDF"].inputs[0].default_value=color_over1
        nodes["Principled BSDF"].inputs[21].default_value=color_over1[3]
        nodes["Principled BSDF"].inputs[7].default_value=0
        nodes["Principled BSDF"].inputs[12].default_value=1
        nodes["Principled BSDF"].inputs[9].default_value=1
        nodes["Principled BSDF"].inputs[4].default_value=0.1
        
        nodes.new("ShaderNodeTexWave")
        nodes["Wave Texture"].inputs[1].default_value=wave_scale
        nodes["Wave Texture"].inputs[2].default_value=wave_distortion
        nodes["Wave Texture"].bands_direction='X'
        
        nodes.new("ShaderNodeTexWave")
        nodes["Wave Texture.001"].inputs[1].default_value=wave_scale
        nodes["Wave Texture.001"].inputs[2].default_value=wave_distortion
        nodes["Wave Texture.001"].bands_direction='Y'
        
        nodes.new("ShaderNodeMixRGB")
        
        nodes.new("ShaderNodeTexNoise")
        nodes["Noise Texture"].inputs[2].default_value=50
        
        nodes.new("ShaderNodeTexNoise")
        nodes["Noise Texture.001"].inputs[2].default_value=noise_scale
        nodes["Noise Texture.001"].inputs[5].default_value=noise_distortion
        
        nodes.new("ShaderNodeValToRGB")
        nodes["ColorRamp"].color_ramp.interpolation = 'EASE'
        nodes["ColorRamp"].color_ramp.elements[0].position = 0
        nodes["ColorRamp"].color_ramp.elements[0].color = color_under
        nodes["ColorRamp"].color_ramp.elements[1].position = 0.75
        nodes["ColorRamp"].color_ramp.elements[1].color = color_over1

        nodes.new("ShaderNodeValToRGB")
        nodes["ColorRamp.001"].color_ramp.interpolation = 'EASE'
        nodes["ColorRamp.001"].color_ramp.elements[0].position = 0
        nodes["ColorRamp.001"].color_ramp.elements[0].color = color_under
        nodes["ColorRamp.001"].color_ramp.elements[1].position = 0.75
        nodes["ColorRamp.001"].color_ramp.elements[1].color = color_over2
        
        nodes.new("ShaderNodeTexCoord")

        if pattern == "checker":
            nodes.new("ShaderNodeTexChecker")
            nodes["Checker Texture"].inputs[3].default_value = checker_scale
        if pattern == "brick":
            nodes.new("ShaderNodeTexBrick")
            nodes["Brick Texture"].inputs[4].default_value = checker_scale
            nodes["Brick Texture"].inputs[5].default_value = 0
            nodes["Brick Texture"].name = "Checker Texture"
        if pattern == "none":
            nodes.new("ShaderNodeTexChecker")
            nodes["Checker Texture"].inputs[3].default_value = 0
        
        nodes.new("ShaderNodeBump")
        nodes["Bump"].inputs[0].default_value=bump_wave #0.4
        
        nodes.new("ShaderNodeBump")
        nodes["Bump.001"].inputs[0].default_value=bump_noise1 #0.2
        
        nodes.new("ShaderNodeBump")
        nodes["Bump.002"].inputs[0].default_value=bump_noise2 # 0.3
        
        links = mat.node_tree.links
        links.new(nodes["Texture Coordinate"].outputs[2], nodes["Wave Texture"].inputs[0])
        links.new(nodes["Texture Coordinate"].outputs[2], nodes["Wave Texture.001"].inputs[0])
        links.new(nodes["Texture Coordinate"].outputs[2], nodes["Noise Texture"].inputs[0])
        links.new(nodes["Texture Coordinate"].outputs[2], nodes["Noise Texture.001"].inputs[0])
        links.new(nodes["Texture Coordinate"].outputs[2], nodes["Checker Texture"].inputs[0])
        links.new(nodes["Wave Texture"].outputs[0], nodes["Mix (Legacy)"].inputs[1])
        links.new(nodes["Wave Texture.001"].outputs[0], nodes["Mix (Legacy)"].inputs[2])
        links.new(nodes["Mix (Legacy)"].outputs[0], nodes["ColorRamp"].inputs[0])
        links.new(nodes["Mix (Legacy)"].outputs[0], nodes["ColorRamp.001"].inputs[0])
        links.new(nodes["ColorRamp"].outputs[0], nodes["Checker Texture"].inputs[1])
        links.new(nodes["ColorRamp.001"].outputs[0], nodes["Checker Texture"].inputs[2])
        links.new(nodes["Checker Texture"].outputs[0], nodes["Principled BSDF"].inputs[0])
        links.new(nodes["Checker Texture"].outputs[0], nodes["Principled BSDF"].inputs[3])
        links.new(nodes["Mix (Legacy)"].outputs[0], nodes["Bump"].inputs[2])
        links.new(nodes["Bump"].outputs[0], nodes["Bump.001"].inputs[3])
        links.new(nodes["Noise Texture"].outputs[0], nodes["Bump.001"].inputs[2])
        links.new(nodes["Bump.001"].outputs[0], nodes["Bump.002"].inputs[3])
        links.new(nodes["Noise Texture.001"].outputs[0], nodes["Bump.002"].inputs[2])
        links.new(nodes["Bump.002"].outputs[0], nodes["Principled BSDF"].inputs[22])
    return



def setup(sequence_name, participant, scene_data, config, texture_params={}):

    # Find the start and end of the video
    final_frame=0
    blankets=[blanket for blanket in bpy.data.objects if "Plane" in blanket.name]
    for blanket in blankets:
        if final_frame<blanket.modifiers["clothMod"].point_cache.frame_end:
            final_frame=blanket.modifiers["clothMod"].point_cache.frame_end
        changeMaterial(ob=blanket, **texture_params)
    start_frame=scene_data["setting_frames"]
    renderParams()

    path_to_save=config["path_to_save_render"]+participant+"/"
    try:
        os.mkdir(path_to_save)
    except OSError as error:
        print(error)  
    path_to_save=path_to_save+sequence_name+"/"
    try:
        os.mkdir(path_to_save)
    except OSError as error:
        print("SKIPPING!",error, flush=True) 
        return -1

    return path_to_save




def renderBlanket(blanket, camera, blanket_number, path_to_save, scene_data, render_params={}):
    path_blanket_video=path_to_save+"blanket_"+str(blanket_number)+"/"+camera.name+"/"
    renderParams(render_params=render_params)
    try:
        os.mkdir(path_to_save+"blanket_"+str(blanket_number)+"/")
    except OSError as error:
        print(error)   
    try:
        os.mkdir(path_blanket_video)
    except OSError as error:
        print(error)  
    
    bpy.data.scenes[0].frame_start = blanket.modifiers["clothMod"].point_cache.frame_start+scene_data["setting_frames"]
    bpy.data.scenes[0].frame_end = blanket.modifiers["clothMod"].point_cache.frame_end
    
    bpy.data.scenes[0].camera = camera
    print()
    c = scene_data["cameras_intrinsics"][camera.name]["c"]

    blanket.hide_render=False
    bpy.data.scenes[0].render.filepath = path_blanket_video
    bpy.ops.render.render(animation=True, scene=bpy.data.scenes[0].name)
    blanket.hide_render=True

    webps=[]
    for (dirpath, dirnames, filenames) in walk(path_blanket_video):
        for i in filenames:
            if ".webp" in i:
                webps.extend([dirpath+"/"+i])
    for webp in webps:
        num=(int(webp.split("/")[-1].split(".")[0])-scene_data["setting_frames"])
        os.rename(webp, path_blanket_video+"image_"+f"{num:05d}"+".webp")

    vid_info=copy.deepcopy(scene_data)
    vid_info["frame_start"]=blanket.modifiers["clothMod"].point_cache.frame_start
    vid_info["frame_end"]=blanket.modifiers["clothMod"].point_cache.frame_end-scene_data["setting_frames"]

    with open(path_blanket_video+"vid_info.json", "w") as f:
        json.dump(vid_info, f)

    return
    


if __name__=="__main__":
    start=time.time()

    bpy.context.preferences.filepaths.save_version=0

    with open("config_render.json", "r") as f:
        config=json.load(f)
        path_to_sim=config["path_to_save_sim"]
        path_to_render=config["path_to_save_render"]

    # These were the parameters used in BlanketGen2-Fit3D
    render_params={
        "max_samples":100,
        "use_denoising":False,
        "use_adaptive_sampling":False,
        "min_samples":100,
    }
    texture_params={
                    "color_over1":[0.9,0.9,0.9,1],
                    "color_over2":[0.1,0.1,0.1,1],
                    "noise_scale":30,
                    "checker_scale":10,
                    "pattern":"checker",
                    }
    patterns = ["checker", "brick", "none"]
    randomize = False

    blends=[]
    for (dirpath, dirnames, filenames) in walk(path_to_sim):
        for i in filenames:
            if ".blend" in i:
                blends.extend([dirpath+"/"+i])
    for blend in blends:
        if os.path.isfile(blend[:-len(blend.split("/")[-1])]+"scene_data.json")==False:
            continue
        with open(blend[:-len(blend.split("/")[-1])]+"scene_data.json", "r") as f:
            scene_data=json.load(f)
        bpy.ops.wm.open_mainfile(filepath=blend)
        # Randomize blanket texture
        if randomize:
            texture_params["color_over1"][0]=0.15+np.random.random()*0.7
            texture_params["color_over1"][1]=0.15+np.random.random()*0.7
            texture_params["color_over1"][2]=0.15+np.random.random()*0.7

            texture_params["color_over2"][0]=0.1*texture_params["color_over1"][0]+np.random.random()*0.9
            texture_params["color_over2"][1]=0.1*texture_params["color_over1"][1]+np.random.random()*0.9
            texture_params["color_over2"][2]=0.1*texture_params["color_over1"][2]+np.random.random()*0.9

            texture_params["checker_scale"]=np.random.random()*15
            texture_params["pattern"]=patterns[int(np.floor(np.random.random()*3))]

        print(texture_params)
        path_to_save=setup(sequence_name=blend.split("/")[-2], participant=blend.split("/")[-3], scene_data=scene_data, config=config, texture_params=texture_params)
        if path_to_save==-1:
            continue
        blankets=[blanket for blanket in bpy.data.objects if "Plane" in blanket.name]
        cameras=[camera for camera in bpy.data.objects if "CAMERA" in camera.type]
        for blanket_ind in range(len(blankets)):
            for camera_ind in range(len(cameras)):
                renderBlanket(blankets[blanket_ind],cameras[camera_ind], blanket_ind, path_to_save, scene_data, render_params=render_params)

    print("TOOK:", time.time()-start)