import os
def extract_frames(inputData_folder):
    inputData = os.listdir(inputData_folder+"videos/")
    for file in inputData:
        # file = file.replace(" ","\ ")
        video_file = inputData_folder+"videos/"+file
        video = f'"{video_file}"'
        print(video)
        name = file.split(".")[0].replace(" ","_")
        output_folder = inputData_folder+"frames_per_sec/"+name
        os.makedirs(output_folder, exist_ok=True)
        # os.system(f'ffmpeg -i {video} -vf "select=not(mod(n\\,30))" -vsync vfr -q:v 1 {output_folder}/_%03d.jpg')
        os.system(f'ffmpeg -i {video} -r 1 -vsync vfr -q:v 1 {output_folder}/_%03d.jpg')
    

if __name__ =="__main__":
    extract_frames("Surgical_videos_example/")
