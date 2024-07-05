# Semi Automatic Labelling with Ultralytics and Label Studio

## Quickstart
Before using this repository, make sure you are already running label studio. You can see the [Installation Docs](https://labelstud.io/) for the installation. I recommend you to run it using Docker. Here is what command I used for running the label studio.

```bash
mkdir -p ~/Documents/Personal/label-studio
cd ~/Documents/Personal/label-studio
```
You change change the path according to your preferences.

```bash
docker run -it --user root -p {PORT}:8080 -v $(pwd)/mydata:/label-studio/data --env LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true --env LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files -v $(pwd)/myfiles:/label-studio/files heartexlabs/label-studio:latest label-studio
```

Note: Change value of `PORT` based on your preferences. The command above will mount your local storage to label studio container, check out this [Sync to Local Storage](https://labelstud.io/guide/storage#Local-storage) for more information.

Once the program is succesfully running, you can go to `http://localhost:{PORT}`

## Running Machine Learning Backend

Please, make sure you already running the label studio, then

1. Clone the repo
```bash
https://github.com/Kecilin-Team/label-studio-semi-automatic.git

cd label-studio-semi-automatic
```

2. Edit the .env file

```bash
cp .env-examples .env
```

- `BACKEND_PORT` is the port that you want to used to run ML Backend
- `LABEL_STUDIO_BASEURL` is local url label studio that is already running. e.g (`192.168.42.42:{PORT}`)
- `LABEL_STUDIO_API_TOKEN` is API token for your label studio
- `MODEL_DIR` the directory that contains `.pt` and `.yaml` file. I recommend you to make a directory in `weights` folder. Example: `weights/person` will consists `<name>.pt` and `<name>.yaml`

Note: 
- If you are running the ML backend in Docker, `LABEL_STUDIO_URL` can’t contain localhost or 0.0.0.0. Use the full IP address instead, e.g. `192.168.42.42`. You can get this using the ifconfig (Unix) or ipconfig (Windows) commands.
- The content of `.yaml` file have to look like this
```
names:
    - person
    - class_2
    - class_3
    .
    .
    .
```
- 


2. Build and start machine learning backend

```bash
sudo docker-compose up -d --build
```

Check if it works:

```bash
$ curl http://localhost:{BACKEND_PORT}/health

{"model_dir":"weights/person","status":"UP","v2":false}
```

## Create Project and Import Dataset

1. Create a Project

I assume you know and already made a project in Label Studio, simply just click the `Create` button on top-left of you label studio UI, then set up your setting such as labels (don't forget to add label names), and also type of annotation which is **Object Detection with Bounding Box**.


2. Import Dataset

You can simply import the dataset by click `Go to import` then drag your dataset. But, this method can't be done if your dataset is big. So, I recommend you to maybe synchronize the label studio to local storage or using cloud storage service. Now, I'll show you how to synchronize label studio into your local storage.

Go to Settings >> Cloud Storage >> Add Source Storage >> Storage Type >> Local Storage

Then, add your storage title (optional), and absolute local path. Remember in the Quickstart, I already add a command where your local storage will be mount to label studio container, check out this `LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/files` part.

So, if before your ran the quickstart command in `~/Documents/Personal/label-studio`, then, the `myfiles` folder will be created inside of that directory. You can move or copy a folder that contains all you images to annotate inside `myfiles`. Example, we have a dataset contains of images named `person_dataset`, then, copy the `person_dataset` into `~/Documents/Personal/label-studio/myfiles`. At the end you will have a directory `~/Documents/Personal/label-studio/myfiles/person_dataset` in your local that contains a bunch of images.

Then, add your **Absolute local path** to be `/label-studio/files/person_dataset`, activate the toggle button, and Check connection. Finally, you can click **Sync Storage**

Note: See quickstart command where there's a mounting process to `label-studio/files` from your local storage.


## Connect to Machine Learning Backend

Go to Settings >> Model >> Enter Name >> Enter Backend URL >> Activate Interactive preannotations >> Validate and Save.

Note: If you are running the ML backend in Docker, backend url can’t contain localhost or 0.0.0.0. Use the full IP address instead, e.g. `192.168.42.42`. You can get this using the ifconfig (Unix) or ipconfig (Windows) commands. So, the backend url will be something like this `http://192.168.100.79:{BACKEND_PORT}`

Then, setup you annotations settings

Go to Settings >> Annotations >> Activate Show before labeling >> Activate Use predictions to prelabel tasks >> Select which prediction that you want to use >> Save



## Enjoy Your Annotation

Once all the steps already completed, you can go back to the project and start your annotation. 