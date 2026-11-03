import bpy
import pickle
import smplx
import numpy as np
import torch
import numpy as np
import bmesh
import os
import time
import json
import itertools
import addon_utils
import PIL.Image

smplx_cfg = {'ext': 'npz',
             'extra_joint_path': '',
             'folder': 'transfer_data/body_models',
             'gender': 'neutral',
             'joint_regressor_path': '',
             'model_type': 'smplx',
             'num_expression_coeffs': 10,
             'smplx': {'betas': {'create': True, 'num': 10, 'requires_grad': True},
                       'body_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'expression': {'create': True, 'num': 10, 'requires_grad': True},
                       'global_rot': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'jaw_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'left_hand_pose': {'create': True,
                                          'pca': {'flat_hand_mean': False, 'num_comps': 12},
                                          'requires_grad': True,
                                          'type': 'aa'},
                       'leye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'reye_pose': {'create': True, 'requires_grad': True, 'type': 'aa'},
                       'right_hand_pose': {'create': True,
                                           'pca': {'flat_hand_mean': False,
                                                   'num_comps': 12},
                                           'requires_grad': True,
                                           'type': 'aa'},
                       'translation': {'create': True, 'requires_grad': True}},
             'use_compressed': False,
             'use_face_contour': True}


# 
# Simulates the cloth up until the specified frame and returns the distance between the closest vertices of the person and the blanket
# This is the slowest part
# 
def closestToPerson(person_obj, cloth_obj, frame):
    bpy.context.scene.frame_set(frame)
    override={"point_cache":cloth_obj.modifiers["clothMod"].point_cache}
    bpy.ops.ptcache.bake(override,bake=False)
    bpy.context.view_layer.update()
    deps = bpy.context.evaluated_depsgraph_get()
    cloth_bmesh = bmesh.new()
    cloth_bmesh.from_object(cloth_obj, deps)
    ret=1000000
    inv=person_obj.matrix_world.inverted()
    for v_ind,vertex in enumerate(cloth_bmesh.verts):
        if v_ind%3 != 0:
            continue
        glob_vert=cloth_obj.matrix_world@vertex.co
        hit, loc, norm, face_index=person_obj.closest_point_on_mesh(inv@glob_vert)
        closest=person_obj.matrix_world@loc
        distance=np.linalg.norm(closest-glob_vert)
        if distance<ret:
            ret=distance
    return ret


# 
# Creates a blanket object (subdivided plane with cloth modifiers), places it in the scene and returns it
# 
def createBlanket(scale=(1,1,1), location=(0,0,5), rotation=(0,0,0), color=(0.25,0.75,0.25,1), number_cuts=50, frame_start=0, frame_end=60, 
                quality=10, collision_quality=10, collision_distance=0.001, thickness=0.003, wave_distortion=7, noise_scale=50, 
                noise_distortion=7, subdiv_levels=2):
    bpy.ops.mesh.primitive_plane_add(location=location, rotation=rotation, scale=(1,1,1))
    bpy.context.object.scale=(scale[0]/2,scale[1]/2,scale[2])
    bpy.context.object.location=location

    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=number_cuts)
    bpy.ops.object.mode_set(mode = 'OBJECT')

    bpy.context.scene.frame_current = frame_start
    bpy.context.object.modifiers.new(name="clothMod",type="CLOTH")
    bpy.context.object.modifiers.new(name="subdivideMod",type="SUBSURF")
    bpy.context.object.modifiers["subdivideMod"].levels = subdiv_levels
    bpy.context.object.modifiers["subdivideMod"].render_levels = subdiv_levels

    bpy.ops.object.shade_smooth()

    bpy.context.object.modifiers["clothMod"].settings.quality = quality
    bpy.context.object.modifiers["clothMod"].collision_settings.collision_quality = collision_quality
    bpy.context.object.modifiers["clothMod"].point_cache.frame_start = frame_start
    bpy.context.object.modifiers["clothMod"].point_cache.frame_end = frame_end
    bpy.context.object.modifiers["clothMod"].collision_settings.use_self_collision = True
    bpy.context.object.modifiers["clothMod"].collision_settings.distance_min = collision_distance
    bpy.context.object.modifiers["clothMod"].collision_settings.self_distance_min = collision_distance if collision_distance<0.5*min(scale)/number_cuts else 0.5*min(scale)/number_cuts
    bpy.context.object.modifiers["clothMod"].settings.time_scale = 1

    return bpy.context.object


# 
# Loads everything into the scene
# 
def setScene(path_to_dataset,subject,action,model,path_to_save,person_friction,bed_friction,is_vertical):
    # Set params
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.view.show_splash=False
    setting_frames = fps*setting_time 
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_start = setting_frames
    

    # Load cameras
    camera_paths = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_dataset+"train/"+subject+"/camera_parameters/"):
        for fname in filenames:
            if fname == action+".json":
                camera_paths.append((dirpath+"/"+fname))
    cameras=[]
    cameras_intrinsics={}

    
    for cam_path in camera_paths:
        with open(cam_path, "r") as f:
            cam_raw = json.load(f)

        Tmatrix = np.zeros(shape=(4,4))
        Tmatrix[:3,:3] = cam_raw["extrinsics"]["R"]
        Tmatrix[3,:3] = cam_raw["extrinsics"]["T"][0]
        Tmatrix[3,3] = 1
        Tmatrix[1:3,0:3] *= -1
        
        bpy.ops.object.camera_add(location=[0,0,0], rotation=[0,0,0])
        bpy.context.scene.camera = bpy.context.object
        bpy.context.scene.render.film_transparent = True

        bpy.context.object.name = cam_path.split("/")[-2]

        fx = cam_raw["intrinsics_wo_distortion"]["f"][0]
        fy = cam_raw["intrinsics_wo_distortion"]["f"][1]
        cx = cam_raw["intrinsics_wo_distortion"]["c"][0]
        cy = cam_raw["intrinsics_wo_distortion"]["c"][1]
        bpy.context.scene.camera.data.sensor_height = 24
        bpy.context.scene.camera.data.sensor_fit = "VERTICAL"
        bpy.context.scene.render.resolution_x = 900
        bpy.context.scene.render.resolution_y = 900
        bpy.context.scene.camera.data.lens = 12*fy/cy
        bpy.context.scene.render.pixel_aspect_y = fx/fy  
        bpy.context.scene.camera.matrix_world = Tmatrix 
        bpy.context.scene.camera.data.shift_x = 0.5-cx/900
        bpy.context.scene.camera.data.shift_y = -0.5+cy/900
        cameras.append(bpy.context.object)
        cameras_intrinsics[bpy.context.scene.camera.name]=cam_raw["intrinsics_wo_distortion"]
    # Create lights
    bpy.ops.object.light_add(type='AREA', align='WORLD', location=(6, 0, 5), rotation=(0,np.pi/2,0))
    bpy.context.object.scale=(2, 4, 2)
    bpy.context.object.data.energy=100
    bpy.ops.object.light_add(type='AREA', align='WORLD', location=(-6, 0, 5), rotation=(0,-np.pi/2,0))
    bpy.context.object.scale=(2, 4, 2)
    bpy.context.object.data.energy=100
    bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, 6, 5), rotation=(-np.pi/2,0,0))
    bpy.context.object.scale=(4, 2, 2)
    bpy.context.object.data.energy=100
    bpy.ops.object.light_add(type='AREA', align='WORLD', location=(0, -6, 5), rotation=(np.pi/2,0,0))
    bpy.context.object.scale=(4, 2, 2)
    bpy.context.object.data.energy=100


    # Load model and animate with shape keys (specified start and end frames)
    with open(path_to_dataset+"train/"+subject+"/smplx/"+action+".json","r") as f:
        smplx_params = json.load(f)
        smplx_params = {key: torch.from_numpy(np.array(smplx_params[key]).astype(np.float32)) for key in smplx_params}
    person_smplx = model(**smplx_params)
    vertices=person_smplx.vertices[0].detach().cpu().numpy().squeeze()
    faces=model.faces
    frame_end=len(person_smplx.vertices)+setting_frames
    bpy.context.scene.frame_end = frame_end

    person_mesh=bpy.data.meshes.new("person_mesh"+str(0))
    person_object=bpy.data.objects.new("person_obj"+str(0), person_mesh)
    person_object.modifiers.new(name="col_mod",type="COLLISION")
    person_object.collision.cloth_friction = person_friction
    person_mesh.from_pydata(vertices, [], faces)
    person_mesh.update()
    bpy.context.collection.objects.link(person_object)
    person_object.is_holdout = True
    person_object.collision.thickness_outer = 0.005
    person_object.shape_key_add(name="Base", from_mix=False)

    pelvis = person_smplx.joints[:, 0, :].numpy()
    transl = smplx_params["transl"]
    global_orient = smplx_params["global_orient"]

    with open(path_to_dataset+"train/"+subject+"/joints3d_25/"+action+".json","r") as f:
        j3d = np.array(json.load(f)["joints3d_25"])
    
    person_axes=[]
    farthest_ds=[]
    upside_down = False
    for frame in (range(setting_frames,frame_end)):
        # Assuming heels are planted in bed
        person_y = np.cross(j3d[frame-setting_frames][6] - j3d[frame-setting_frames][0], j3d[frame-setting_frames][3] - j3d[frame-setting_frames][0])
        person_y[2] = 0
        person_y = person_y / np.linalg.norm(person_y)
        feet_avg = (j3d[frame-setting_frames][6] + j3d[frame-setting_frames][3]) / 2
        person_z = (0,0,1)
        person_x = np.cross(person_z, person_y)
        if upside_down:
            person_x *= -1
            person_z *= -1
            person_y *= -1
        person_axes.append((person_x,person_y,person_z))

        print("Loading body on frame:", frame-setting_frames)
        vertices=person_smplx.vertices[frame-setting_frames].detach().cpu().numpy().squeeze()
        farthest_d=0
        person_rotation_matrix=np.array(person_object.matrix_world)[:3,:3]
        person_y_vertex_space=person_y@person_rotation_matrix
        for vertex in range(len(person_object.data.vertices)):
            person_object.data.vertices[vertex].co.xyz=vertices[vertex]
            d=np.dot(vertices[vertex], person_y_vertex_space)
            if d < farthest_d:
                farthest_d=d
        farthest_ds.append(farthest_d)
        person_object.shape_key_add(name=str(frame), from_mix=False)

        person_object.data.shape_keys.key_blocks[str(frame)].value = 0
        person_object.data.shape_keys.key_blocks[str(frame)].keyframe_insert(data_path="value", frame=0)
        person_object.data.shape_keys.key_blocks[str(frame)].value = 0
        person_object.data.shape_keys.key_blocks[str(frame)].keyframe_insert(data_path="value", frame=frame-1)
        person_object.data.shape_keys.key_blocks[str(frame)].value = 1
        person_object.data.shape_keys.key_blocks[str(frame)].keyframe_insert(data_path="value", frame=frame)
        person_object.data.shape_keys.key_blocks[str(frame)].value = 0
        person_object.data.shape_keys.key_blocks[str(frame)].keyframe_insert(data_path="value", frame=frame+1)
        person_object.keyframe_insert(data_path="location", frame=frame)
        person_object.keyframe_insert(data_path="rotation_euler", frame=frame) 
  
    # Add bed
    bpy.ops.mesh.primitive_cube_add(location = (0,0,-1))
    bed=bpy.context.object
    bpy.context.object.scale = (5,5,0.5)
    bed.hide_render=True
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.subdivide(number_cuts=10)
    bpy.ops.object.mode_set(mode = 'OBJECT')
    bed.modifiers.new(name="col_mod",type="COLLISION")
    bed.collision.cloth_friction = bed_friction
    # Reposition the bed to be behind the person
    if is_vertical:
        for frame in range(frame_end):
            actualframe = max(0,frame-setting_frames)
            for i in range(3):
                for j in range(3):
                    bed.matrix_world[i][j]=person_axes[actualframe][i][j]
            bed.location=[person_axes[actualframe][1][i]*(farthest_ds[actualframe]-0.5-0.1) for i in range(3)]
            bpy.context.object.scale = (5,0.5,5)


            bed.keyframe_insert(data_path="location", frame=frame)
            bed.keyframe_insert(data_path="rotation_euler", frame=frame) 



    person_axes=np.array(person_axes)

    # Save json with scene info
    datajson = {}
    datajson["setting_frames"] = setting_frames
    datajson["resolution"] = (900,900)
    datajson["blanket_shape"] = blanket_shape
    datajson["blanket_location"] = blanket_location
    datajson["person_friction"] = person_friction
    datajson["blanket_n_cuts"] = blanket_n_cuts
    datajson["blanket_quality"] = blanket_quality
    datajson["blanket_collision_quality"] = blanket_collision_quality
    datajson["blanket_collision_distance"] = blanket_collision_distance
    datajson["setting_frames"] = setting_frames
    datajson["subdiv_levels"] = subdiv_levels
    datajson["cameras_intrinsics"] = cameras_intrinsics
    with open(path_to_save+subject+"/"+action+"/scene_data.json", "w") as f:
        json.dump(datajson, f)
    return person_axes, person_object, j3d


# 
# Generates the next blanket
# 
def generateNextBlanket(path_to_dataset,subject,action,
            blanket_shape,blanket_location,person_friction,bed_friction,
            blanket_n_cuts,blanket_quality,blanket_collision_quality,
            blanket_collision_distance,frame_start,
            setting_time,color,fps,timeout,subdiv_levels,skip,person_axes,person,j3d,is_vertical):#,fractions_person):
    

    # Figure out when to start the blanket sim
    bpy.context.scene.frame_current = frame_start
    setting_frames = setting_time*fps
    og_frame_start = max(0,frame_start - setting_frames)
    frame_end=bpy.context.scene.frame_end
    # Figure out where to drop the blanket
    print(len(person_axes), og_frame_start)
    if not is_vertical:
        blanket_location_absolute = np.array(blanket_location)@((0,1,0),(0,0,1),(1,0,0)) + j3d[og_frame_start][0]
    else:
        blanket_location_absolute = np.array(blanket_location)@person_axes[og_frame_start] + j3d[og_frame_start][0]
    blanket=createBlanket(frame_start=frame_start,frame_end=frame_end, scale=blanket_shape, 
    location=blanket_location_absolute,number_cuts=blanket_n_cuts,quality=blanket_quality,
    collision_quality=blanket_collision_quality,collision_distance=blanket_collision_distance, 
    subdiv_levels=subdiv_levels)
    if is_vertical:
        blanket_center_vector = j3d[og_frame_start][0] - blanket_location_absolute
        blanket_center_cross = np.cross(np.array((0,0,1)), blanket_center_vector/np.linalg.norm(blanket_center_vector))
        blanket_angle = np.arcsin(np.linalg.norm(blanket_center_cross))
        blanket.rotation_mode = 'AXIS_ANGLE'
        bpy.context.object.rotation_axis_angle[0] = blanket_angle
        bpy.context.object.rotation_axis_angle[1],bpy.context.object.rotation_axis_angle[2], bpy.context.object.rotation_axis_angle[3] = blanket_center_cross

    # Simulate and bake the blanket
    final_frame=bpy.context.scene.frame_end
    if is_vertical:
        for frame in range(frame_start,frame_end):
            frame_to_use = 0
            person_gravity = (0,-1,0)
            absolute_gravity = np.array((person_gravity[0]*person_axes[frame_to_use][0][0] + person_gravity[1]*person_axes[frame_to_use][1][0] + person_gravity[2]*person_axes[frame_to_use][2][0],
                                        person_gravity[0]*person_axes[frame_to_use][0][1] + person_gravity[1]*person_axes[frame_to_use][1][1] + person_gravity[2]*person_axes[frame_to_use][2][1],
                                        person_gravity[0]*person_axes[frame_to_use][0][2] + person_gravity[1]*person_axes[frame_to_use][1][2] + person_gravity[2]*person_axes[frame_to_use][2][2]))
            bpy.context.scene.gravity=9.81*absolute_gravity
            bpy.context.scene.keyframe_insert(data_path="gravity", frame=frame)
    

    for frame in range(frame_start,frame_end):
        start=time.time()
        r=closestToPerson(person,blanket,frame)

        if frame < setting_frames+frame_start:
            continue
        if time.time()-start>timeout:
            print("Sim timed out at frame:", frame)
            final_frame=frame
            blanket.modifiers["clothMod"].point_cache.frame_end = frame
            break
        if r > 0.03:
            print("Blanket too far away at frame:", frame)
            blanket.modifiers["clothMod"].point_cache.frame_end = frame
            final_frame=frame
            break
    

    override={"point_cache":blanket.modifiers["clothMod"].point_cache}
    bpy.ops.ptcache.bake_from_cache(override)


    # Hide blanket on render
    blanket.hide_render=True
    # Return the frame where sim ended, -1 if done
    ret=max(final_frame-setting_frames, frame_start+skip)
    if ret>=bpy.context.scene.frame_end-setting_frames:
        print("Finished cloth sims")
        ret=-1
    return ret

def main(path_to_save, path_to_dataset, smpl_path, is_vertical_dict, split=None,
    blanket_shape=(2,2,1),blanket_location=(0,-0.5,1), person_friction=15, bed_friction=15,
    blanket_n_cuts=50, blanket_quality=20, blanket_collision_quality=15, blanket_collision_distance=0.001,
    sun_orientation=(0,0,0), setting_time=1, color=(0.25,0.75,0.25,1), fps=50, timeout=60, subdiv_levels=2, skip=24):
    # Load the dataset info 
    with open(path_to_dataset+"fit3d_info.json", "r") as f:
        info = json.load(f)

    # Define structure of dataset
    person_seqs = []
    for (dirpath, dirnames, filenames) in os.walk(path_to_dataset):
        for fname in filenames:
            if "smplx" in dirpath:
                person_seqs.append((dirpath.split("/")[-2], fname[:-5]))

    # Configure split (for generating with multiple instances)
    if split is not None:
        pass

    # Load SMPL-X model
    global smplx_cfg
    smplxNeutral = smplx.build_layer(smpl_path, **smplx_cfg)

    # For each original video
    for seq in person_seqs:
        is_vertical = is_vertical_dict[seq[1]]
        if is_vertical == "True":
            is_vertical = True
        elif is_vertical == "False":
            is_vertical = False
        else:
            continue
        try:
            os.mkdir(path_to_save)
        except OSError as error:
            print(error) 
        try:
            os.mkdir(path_to_save+seq[0])
        except OSError as error:
            print(error) 
        try:
            os.mkdir(path_to_save+seq[0]+"/"+seq[1])
        except OSError as error:
            print("SKIPPED!",error, flush=True) 
            continue
            
        person_axes, person, j3d = setScene(path_to_dataset=path_to_dataset,subject=seq[0],action=seq[1],model=smplxNeutral,
            path_to_save=path_to_save,person_friction=person_friction,bed_friction=bed_friction,is_vertical=is_vertical)

        # Skip till person isn't standing up
        frame=0
        if not is_vertical:
            while True:
                temp_frame = max(0,frame-setting_time*fps)
                feet_avg = (j3d[temp_frame][6] + j3d[temp_frame][3]) / 2
                temp_z = j3d[temp_frame][0] - feet_avg
                print("Skipping frame:", frame, "Z:", temp_z, flush=True)
                if frame>=len(person_axes):
                    print("WARNING: Skipped all the frames!!!")
                    break
                upside_down = False
                if upside_down:
                    if temp_z[2]>-0.25:
                        break
                else:
                    if temp_z[2]<0.25:
                        break
                frame+=1

        #   while not finished
        blanket_count = 0
        while frame>=0:
            temp_frame = max(0,frame-setting_time*fps)
            feet_avg = (j3d[temp_frame][6] + j3d[temp_frame][3]) / 2
            temp_z = j3d[temp_frame][0] - feet_avg
        #   generate next blanket
            if not is_vertical:
                if upside_down:
                    if temp_z[2]<-0.25:
                        break
                else:
                    if temp_z[2]>0.25:
                        break
            frame = generateNextBlanket(path_to_dataset=path_to_dataset,subject=seq[0],action=seq[1],
            blanket_shape=blanket_shape,blanket_location=blanket_location,person_friction=person_friction,bed_friction=bed_friction,
            blanket_n_cuts=blanket_n_cuts,blanket_quality=blanket_quality,blanket_collision_quality=blanket_collision_quality,
            blanket_collision_distance=blanket_collision_distance,frame_start=frame,setting_time=setting_time,color=color,fps=fps,timeout=timeout,
            subdiv_levels=subdiv_levels,skip=skip, person_axes=person_axes, person=person, j3d=j3d,is_vertical=is_vertical)
            blanket_count += 1
    #   save blend file
        print(f"Blankets generated: {blanket_count}")
        if blanket_count > 10:
            print("WARNING: MORE THAN 10 BLANKETS GENERATED!!!")
        bpy.ops.wm.save_as_mainfile(filepath=path_to_save+seq[0]+"/"+seq[1]+"/"+seq[1]+".blend") 
        print("Saved at", path_to_save+seq[0]+"/"+seq[1]+"/"+seq[1]+".blend")
        break
    return


if __name__=="__main__":

    start=time.time()

    bpy.context.preferences.filepaths.save_version=0

    with open("config.json", "r") as f:
        config = json.load(f)
        path_to_dataset = config["path_to_dataset"]
        smpl_path = config["smpl_path"]
        path_to_save = config["path_to_save_sim"]

    with open("is_vertical.json", "r") as f:
        is_vertical_dict=json.load(f)

    blanket_quality = 20
    blanket_collision_quality = 20
    blanket_collision_distance = 0.005
    blanket_n_cuts = 80
    subdiv_levels = 2
    blanket_shape = (1.6,1.5,1)
    blanket_location = (0,1,-0.3)

    person_friction = 30
    bed_friction = 10
    
    fps = 50
    setting_time = 1

    main(path_to_save=path_to_save,
        smpl_path=smpl_path,
        path_to_dataset=path_to_dataset,
        is_vertical_dict=is_vertical_dict,
        blanket_quality=blanket_quality,
        blanket_collision_quality=blanket_collision_quality,
        blanket_collision_distance=blanket_collision_distance,
        blanket_n_cuts=blanket_n_cuts,
        blanket_shape=blanket_shape,
        blanket_location=blanket_location,
        setting_time=setting_time,
        subdiv_levels=subdiv_levels,
        person_friction=person_friction,
        bed_friction=bed_friction,
        fps=fps, split=None)

    print("\nIn total it took:", time.time()-start, "s")