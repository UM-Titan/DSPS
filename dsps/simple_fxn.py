import cv2, os, toml, shutil, json,glob, random 
from shutil import copyfile
import pandas as pd


def show_annotations(input_folder):
    

    all_files = os.listdir(input_folder)
    for cimg in all_files:
        if cimg.endswith('.jpg'):
            cimg_path = os.path.join(input_folder,cimg)
            cannt_path = os.path.join(input_folder,cimg.replace('.jpg','.txt'))
            img_file__ = cv2.imread(cimg_path);
            annt_file = np.loadtxt(cannt_path)

            img_h, img_w, _ = img_file__.shape
            img_w, img_h
            dims = annt_file.ndim
            if dims > 1:
                for cbox in annt_file:
                    cbox = cbox[1:]
                    
                    x11, y11 = int((cbox[0] - cbox[2]/2)*img_w), int((cbox[1] - cbox[3]/2)*img_h)
                    x22, y22 = int((cbox[0] + cbox[2]/2)*img_w), int((cbox[1] + cbox[3]/2)*img_h)
                    img_file__ = cv2.rectangle(img_file__, (x11, y11), (x22, y22), (0,255,0), 3)
            else:
                cbox = annt_file[1:]
                x11, y11 = int((cbox[0] - cbox[2]/2)*img_w), int((cbox[1] - cbox[3]/2)*img_h)
                x22, y22 = int((cbox[0] + cbox[2]/2)*img_w), int((cbox[1] + cbox[3]/2)*img_h)
                img_file__ = cv2.rectangle(img_file__, (x11, y11), (x22, y22), (0,255,0), 3)
        cv2.imshow('',img_file__)
        cv2.waitKey(0)


def toml2cvac(automl_path,dst, cls):

    if not os.path.isdir(dst):
        os.makedirs(os.path.join(dst,'images'))
    else:
        shutil.rmtree(dst)
        cdst = os.path.join(dst,'images')
        os.makedirs(cdst)
    full_path = os.path.join(automl_path, 'annotations')
    files = os.listdir(full_path)
    annotations = []
    images = []
    segmentation = []
    image_id = 0; det_id = 0; iscrowd = 0;

    categories = []
    for cl in cls:
        cls_id = cls.index(cl)+1
        categories.append({"supercategory": "", "id":cls_id, "name":cl})

    img_added = []
    img_added_ = []
    for curfile in files:
        img_added_.append(curfile)
        # img_added.append(curfile) # delete
        cfilepath = os.path.join(full_path, curfile)
        data_configs = toml.load(cfilepath)
        for data_config in data_configs['objects']:
            w = (data_config['xmax'] - data_config['xmin'])
            h = (data_config['ymax'] - data_config['ymin'])
            area = w*h

            bbox = [data_config['xmin'], data_config['ymin'],
                    w, h]

            category_id = cls.index(data_config['class']) + 1
            cobj = {'segmentation': segmentation, 'category_id': category_id, 'id': det_id, 'area': area,
             'iscrowd': iscrowd, 'bbox': bbox, 'image_id': image_id}
            annotations.append(cobj)
            if not curfile in img_added:
                cimgs = {"flickr_url": "", "id": image_id, "date_captured": 0, "width": data_configs['width'],
                "license": 0, "file_name": curfile.replace('.toml',''),
                "coco_url": "", "height": data_configs['height']}
                images.append(cimgs)
                img_added.append(curfile)

            det_id += 1
        image_id+=1

    df_json = {}
    df_json['info'] = {"contributor": "", "year": "", "description": "", "version": "", "url": "", "date_created": ""}
    df_json['annotations'] = annotations
    df_json['images'] = images
    df_json['categories'] = categories
    df_json['licenses'] = [{"url": "", "id": 0, "name": ""}]
    with open(dst + '/demo.json', 'w') as fp:
        json.dump(df_json, fp)

    # ## copy images
    full_path = os.path.join(automl_path, 'images')
    img_files = os.listdir(full_path)
    # ccdst = os.path.abspath(dst)
    for img in img_files:
        src_file = os.path.join(full_path,img)
        dst_file = os.path.join(dst, 'images',img)
        
        if os.path.basename(dst_file).replace('.jpg','.toml') in img_added_:
            if not(os.path.isfile(dst_file)):
                shutil.copy(src_file,dst_file)

def cvat2yolo(cfolder,json_file,img_path, output_path, val_path):
    # print (cfolder)
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        os.makedirs(val_path)
        
    with open(json_file) as f:
        json_data = json.load(f)
    
    img_data = json_data['images']
    all_annts = json_data['annotations']
    # print (json_data['categories'])
    

    cls_names = {}
    for catv in json_data['categories']:
      cls_names[catv['id']] = catv['name']
    print (cls_names)

    all_cls = []
    
    cls_indx = list(range(0,len(cls_names)))
    cur_cls = list(cls_names.values())
    print ('total number of classes: {}'.format(len(cur_cls)))
    
    # val_data = random.sample(img_data,int(len(img_data)*0.2))
    # train_data = list(set(img_data).difference(val_data))
    
    random.shuffle(img_data)
    val_data = img_data[0:int(len(img_data)*0.2)]
    train_data = img_data[int(len(img_data)*0.2):]
    
    # print (cur_cls)
    for img in train_data:
        cur_img, cur_id, width, height = img['file_name'],img['id'], img['width'], img['height']
        annt_match = []
        for cur_annt in all_annts:
            if(cur_annt['image_id']) == cur_id:
                annt_match.append(cur_annt)
        full_img_path = os.path.join(img_path,cur_img)
        shutil.copy(full_img_path,os.path.join(output_path,cfolder + '_' + cur_img))
    
        frame = cv2.imread(full_img_path)
        for cur_match in annt_match:
            bbox = cur_match['bbox']
            cls = cls_names[cur_match['category_id']]
            if not cls in all_cls:all_cls.append(cls)
            x1,y1 = int(bbox[0]),int(bbox[1])
            x2, y2 = int(bbox[0])+int(bbox[2]),int(bbox[1]+int(bbox[3]))
    
            dw = 1. / width
            dh = 1. / height
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            file_path_img = os.path.join(output_path,cfolder + '_' + cur_img)
            filename, file_extension = os.path.splitext(file_path_img)
            with open(file_path_img.replace(file_extension,'.txt'), 'a+') as f: 
                f.write(' '.join([str(int(cls_indx[cur_cls.index(cls)])), str(float(x)), str(float(y)), str(float(w)), str(float(h))]))
                f.write('\n')
    
    print (all_cls)
    for img in val_data:
        cur_img, cur_id, width, height = img['file_name'],img['id'], img['width'], img['height']
        annt_match = []
        for cur_annt in all_annts:
            if(cur_annt['image_id']) == cur_id:
                annt_match.append(cur_annt)
        full_img_path = os.path.join(img_path,cur_img)
        shutil.copy(full_img_path,os.path.join(val_path,cfolder + '_' + cur_img))
    
        frame = cv2.imread(full_img_path)
        for cur_match in annt_match:
            bbox = cur_match['bbox']
            cls = cls_names[cur_match['category_id']]
            if not cls in all_cls:all_cls.append(cls)
            x1,y1 = int(bbox[0]),int(bbox[1])
            x2, y2 = int(bbox[0])+int(bbox[2]),int(bbox[1]+int(bbox[3]))
    
            dw = 1. / width
            dh = 1. / height
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            file_path_img = os.path.join(val_path,cfolder + '_' + cur_img)
            filename, file_extension = os.path.splitext(file_path_img)
            with open(file_path_img.replace(file_extension,'.txt'), 'a+') as f: 
                f.write(' '.join([str(int(cls_indx[cur_cls.index(cls)])), str(float(x)), str(float(y)), str(float(w)), str(float(h))]))
                f.write('\n')

def cvat2yolo_(json_file,img_path, output_path, val_path):
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        os.makedirs(val_path)
    with open(json_file) as f:
        json_data = json.load(f)
    
    img_data = json_data['images']
    all_annts = json_data['annotations']
    # print (json_data['categories'])
    

    cls_names = {}
    for catv in json_data['categories']:
      cls_names[catv['id']] = catv['name']
    print (cls_names)

    all_cls = []
    
    cls_indx = list(range(0,len(cls_names)))
    cur_cls = list(cls_names.values())
    print ('total number of classes: {}'.format(len(cur_cls)))
    
    # val_data = random.sample(img_data,int(len(img_data)*0.2))
    # train_data = list(set(img_data).difference(val_data))
    
    random.shuffle(img_data)
    val_data = img_data[0:int(len(img_data)*0.2)]
    train_data = img_data[int(len(img_data)*0.2):]
    
    # print (cur_cls)
    for img in train_data:
        cur_img, cur_id, width, height = img['file_name'],img['id'], img['width'], img['height']
        annt_match = []
        for cur_annt in all_annts:
            if(cur_annt['image_id']) == cur_id:
                annt_match.append(cur_annt)
        full_img_path = os.path.join(img_path,cur_img)
        shutil.copy(full_img_path,os.path.join(output_path,cur_img))
    
        frame = cv2.imread(full_img_path)
        for cur_match in annt_match:
            bbox = cur_match['bbox']
            cls = cls_names[cur_match['category_id']]
            if not cls in all_cls:all_cls.append(cls)
            x1,y1 = int(bbox[0]),int(bbox[1])
            x2, y2 = int(bbox[0])+int(bbox[2]),int(bbox[1]+int(bbox[3]))
    
            dw = 1. / width
            dh = 1. / height
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            file_path_img = os.path.join(output_path,cur_img)
            filename, file_extension = os.path.splitext(file_path_img)
            with open(file_path_img.replace(file_extension,'.txt'), 'a+') as f: 
                f.write(' '.join([str(int(cls_indx[cur_cls.index(cls)])), str(float(x)), str(float(y)), str(float(w)), str(float(h))]))
                f.write('\n')
    
    print (all_cls)
    for img in val_data:
        cur_img, cur_id, width, height = img['file_name'],img['id'], img['width'], img['height']
        annt_match = []
        for cur_annt in all_annts:
            if(cur_annt['image_id']) == cur_id:
                annt_match.append(cur_annt)
        full_img_path = os.path.join(img_path,cur_img)
        shutil.copy(full_img_path,os.path.join(val_path,cur_img))
    
        frame = cv2.imread(full_img_path)
        for cur_match in annt_match:
            bbox = cur_match['bbox']
            cls = cls_names[cur_match['category_id']]
            if not cls in all_cls:all_cls.append(cls)
            x1,y1 = int(bbox[0]),int(bbox[1])
            x2, y2 = int(bbox[0])+int(bbox[2]),int(bbox[1]+int(bbox[3]))
    
            dw = 1. / width
            dh = 1. / height
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            x = x * dw
            w = w * dw
            y = y * dh
            h = h * dh
            file_path_img = os.path.join(val_path,cur_img)
            filename, file_extension = os.path.splitext(file_path_img)
            with open(file_path_img.replace(file_extension,'.txt'), 'a+') as f: 
                f.write(' '.join([str(int(cls_indx[cur_cls.index(cls)])), str(float(x)), str(float(y)), str(float(w)), str(float(h))]))
                f.write('\n')