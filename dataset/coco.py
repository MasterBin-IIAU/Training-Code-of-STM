import os
import os.path as osp
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import torchvision
from torch.utils import data
import random
import glob
import math
import cv2
import json

import imgaug as ia
import imgaug.augmenters as iaa
import math

class sampled_aug(object):
    def __init__(self):
        self.affinity = iaa.Affine(rotate=(-30, 30),shear=(-20, 20),scale={"x": (0.8, 1.2), "y": (0.8, 1.2)})
        self.ela = iaa.ElasticTransformation(alpha = 50, sigma = 5)
    def __call__(self,image,label):
        image,label = self.affinity(image = image,segmentation_maps = label[np.newaxis,:,:,np.newaxis])
        # image,label = self.ela(image = image,segmentation_maps = label)   
        label = label[0,:,:,0]
        return image,label

class Coco_MO_Train(data.Dataset):
    # for multi object, do shuffling

    def __init__(self, img_dir,json_path):
        self.image_dir = img_dir


        self.K = 11
        self.skip = 0
        with open(json_path) as f:
            data = json.load(f)

        self.images_list = data['images']  # length 118287
        self.anno_list = data['annotations']  # length 860001
        self.sampled_aug = sampled_aug()
        #self.augment = PhotometricDistort()
        self.in_sz = 384  # Network Input size during training (default value is 384)

    def __len__(self):
        return len(self.images_list)

    def change_skip(self,f):
        self.skip = f

    def To_onehot(self, mask):
        M = np.zeros((self.K, mask.shape[0], mask.shape[1]), dtype=np.uint8)
        for k in range(self.K):
            M[k] = (mask == k).astype(np.uint8)
        return M
    
    def All_to_onehot(self, masks):
        Ms = np.zeros((self.K, masks.shape[0], masks.shape[1], masks.shape[2]), dtype=np.uint8)
        for n in range(masks.shape[0]):
            Ms[:,n] = self.To_onehot(masks[n])
        return Ms

    def Augmentation(self, image, label,sampled_f_m = None):
        """
        label: (h, w)
        """
        # Scaling
        h,w = label.shape
        # 短边缩放到480
        if w<h:
            factor = 480/w
            image = cv2.resize(image, (480, int(factor * h)), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (480, int(factor * h)), interpolation=cv2.INTER_NEAREST)             
        else:
            factor = 480/h
            image = cv2.resize(image, (int(factor * w), 480), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (int(factor * w), 480), interpolation=cv2.INTER_NEAREST)           

        # Random flipping
        if random.random() < 0.5:
            image = np.fliplr(image).copy()  # HWC
            label = np.fliplr(label).copy()  # HW

        h,w = label.shape

        #affinity
        image1 = image.copy()
        label1 = label.copy()
        image2 = image.copy()
        label2 = label.copy()

        dst_points = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
        tx1 = random.randint(-w//10,w//10)
        ty1 = random.randint(-h//10,h//10)
        tx2 = random.randint(-w//10,w//10)
        ty2 = random.randint(-h//10,h//10)
        tx3 = random.randint(-w//10,w//10)
        ty3 = random.randint(-h//10,h//10)
        tx4 = random.randint(-w//10,w//10)
        ty4 = random.randint(-h//10,h//10)
        src_points = np.float32([[0 + tx1,0 + ty1],[0 + tx2,h-1 + ty2],[w-1 + tx3,h-1 + ty3],[w-1 + tx4,0 + ty4]])
        H1,_ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10)

        dst_points = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]])
        tx1_ = random.randint(0,w//10)
        ty1_ = random.randint(0,h//10)
        tx2_ = random.randint(0,w//10)
        ty2_ = random.randint(0,h//10)
        tx3_ = random.randint(0,w//10)
        ty3_ = random.randint(0,h//10)
        tx4_ = random.randint(0,w//10)
        ty4_ = random.randint(0,h//10)
        src_points = np.float32([[0 + tx1 + tx1_ * tx1/(abs(tx1)+1e-5),0 + ty1 + ty1_ * ty1/(abs(ty1)+1e-5)],[0 + tx2 + tx2_ * tx2/(abs(tx2)+1e-5),h-1 + ty2 + ty2_ * ty2/(abs(ty2)+1e-5)],[w-1 + tx3 + tx3_ * tx3/(abs(tx3)+1e-5),h-1 + ty3 + ty3_ * ty3/(abs(ty3)+1e-5)],[w-1 + tx4 + tx4_ * tx4/(abs(tx4)+1e-5),0 + ty4 + ty4_ * ty4/(abs(ty4)+1e-5)]])
        H2,_ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10)
        # 先对原始图像和标签做仿射变换，得到增强后的图像
        image1 = cv2.warpPerspective(image1, H1, (w,h),flags = cv2.INTER_LINEAR) 
        label1 = cv2.warpPerspective(label1, H1, (w,h),flags = cv2.INTER_NEAREST)   
        image2 = cv2.warpPerspective(image2, H2, (w,h),flags = cv2.INTER_LINEAR) 
        label2 = cv2.warpPerspective(label2, H2, (w,h),flags = cv2.INTER_NEAREST)      
        # 对三张做了数据增强的标签图求了一个并集，找到一个能够把三张图像上所有实例包含在内的矩形
        ob_loc = ((label + label1 + label2) > 0).astype(np.uint8)
        box = cv2.boundingRect(ob_loc)

        x_min = box[0]
        x_max = box[0] + box[2]
        y_min = box[1]
        y_max = box[1] + box[3]

        if x_max - x_min >self.in_sz:
            # 如果目标存在的区域比要裁剪的区域更大，则从中随机取一个384x384的区域即可
            start_w = random.randint(x_min,x_max - self.in_sz)
        elif x_max - x_min == self.in_sz:
            start_w = x_min
        else:
            # 如果目标存在的区域没有384x384那么大，则我们要保证能把整个目标区域完整地包括进来
            start_w = random.randint(max(0,x_max-self.in_sz), min(x_min,w - self.in_sz))

        if y_max - y_min >self.in_sz:
            start_h = random.randint(y_min,y_max - self.in_sz)
        elif y_max - y_min == self.in_sz:
            start_h = y_min
        else:
            start_h = random.randint(max(0,y_max-self.in_sz), min(y_min,h - self.in_sz))
        # Cropping

        end_h = start_h + self.in_sz
        end_w = start_w + self.in_sz

        # 用前面得到的矩形，对三张图像取切片（切片的位置是一致的）
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        image1 = image1[start_h:end_h, start_w:end_w]
        label1 = label1[start_h:end_h, start_w:end_w]
        image2 = image2[start_h:end_h, start_w:end_w]
        label2 = label2[start_h:end_h, start_w:end_w]
        # 如果要粘贴的实例不为空，则要将其粘贴到原生的图像上去
        if sampled_f_m is not None:
            for sf,sm in sampled_f_m:
                h,w = sm.shape
                # frame2相关的值是在frame1基础上加了一个偏移，frame3相关的值则是在frame2基础上加的偏移
                frame1x = random.randint(0,self.in_sz-1)
                frame1y = random.randint(0,self.in_sz-1)
                frame2x = min(self.in_sz-1,max(0,random.randint(0,40) + frame1x))
                frame2y = min(self.in_sz-1,max(0,random.randint(0,20) + frame1y))
                frame3x = min(self.in_sz-1,max(0,random.randint(0,40) + frame2x))
                frame3y = min(self.in_sz-1,max(0,random.randint(0,20) + frame2y))

                image, label = self.copy_paste(sf, sm, frame1y, frame1x, h, w, image, label)

                sf,sm = self.sampled_aug(sf,sm)
                image1, label1 = self.copy_paste(sf, sm, frame2y, frame2x, h, w, image1, label1)

                sf,sm = self.sampled_aug(sf,sm)
                image2, label2 = self.copy_paste(sf, sm, frame3y, frame3x, h, w, image2, label2)


        # 对图像做归一化，变成[0, 1]之间的数
        image = image /255.
        image1 = image1 /255.
        image2 = image2 /255.


        return [image,image1,image2], [label,label1,label2]

    def copy_paste(self, sf, sm, framey, framex, h, w, image, label):
        sf_c = sf.copy()
        sf_c[sm == 0] = 0
        # coords on the original frame (cropped), 384x384
        of_y1 = max(0, framey - h // 2)
        of_y2 = min(self.in_sz, framey + h - h // 2)
        of_x1 = max(0, framex - w // 2)
        of_x2 = min(self.in_sz, framex + w - w // 2)
        # coords on the sampled frame (cropped), dynamic size
        sf_y1 = max(0, h // 2 - framey)
        sf_y2 = min(h, h // 2 + self.in_sz - framey)
        sf_x1 = max(0, w // 2 - framex)
        sf_x2 = min(w, w // 2 + self.in_sz - framex)
        # 这里就是把实例粘贴到当前图像和label上
        label[of_y1:of_y2, of_x1:of_x2] \
            [sm[sf_y1:sf_y2, sf_x1:sf_x2] != 0] = np.max(sm)
        image[of_y1:of_y2, of_x1:of_x2] \
            [sm[sf_y1:sf_y2, sf_x1:sf_x2] != 0] = 0
        image[of_y1:of_y2, of_x1:of_x2] = \
            image[of_y1:of_y2, of_x1:of_x2] + sf_c[sf_y1:sf_y2, sf_x1:sf_x2]
        return image, label

    def mask_process(self,mask,f,num_object,ob_list):
        n = num_object
        mask_ = np.zeros(mask.shape).astype(np.uint8)
        if f == 0:
            for i in range(1,11):
                if np.sum(mask == i) > 350:
                    n += 1
                    ob_list.append(i)
            if n > 5:
                n = 5
                ob_list = random.sample(ob_list,n)
        for i,l in enumerate(ob_list):
            mask_[mask == l] = i + 1
        return mask_,n,ob_list 

    def __getitem__(self, index):
        images = self.images_list[index]
        image_name = images['file_name']
        url = images['coco_url']
        id_ = images['id']
        # 获取该张图片对应的实例信息（包括mask，box等等）
        # TODO: 现在这种写法效率太低，应该先把图片和ID的对应关系预先保存下来，然后直接读取
        instances_list = []
        for anno in self.anno_list:
            if anno['image_id'] == id_:
                instances_list.append(anno)

        info = {}
        info['name'] = image_name
        info['num_frames'] = 3

        N_frames = np.empty((3,)+(self.in_sz,self.in_sz,)+(3,), dtype=np.float32)
        N_masks = np.empty((3,)+(self.in_sz,self.in_sz,), dtype=np.uint8)
        frames_ = []
        masks_ = []

        # print(os.path.join(self.image_dir,image + '.jpg'),os.path.join(self.mask_dir,image + '.png'))
        frame = np.array(Image.open(os.path.join(self.image_dir,image_name)).convert('RGB'))
        h,w,_ = frame.shape
        mask = np.zeros((h,w,20)).astype(np.uint8)

        if random.random() < 0.5:
            object_index = 1
            # mask list是把mask大小满足要求（大于2000个像素）的实例mask加入其中
            mask_list = []
            for inst in instances_list:
                segs = inst['segmentation']
                segs_list = []
                try:
                    for seg in segs:
                        if len(np.array(seg).shape) == 0:
                            continue
                        tmp = np.array(seg).reshape(-1,2).astype(np.int32)
                        segs_list.append(tmp)
                    tmp_mask = np.zeros((h,w))
                    tmp_mask = cv2.fillPoly(tmp_mask, segs_list,1)
                    if np.sum(tmp_mask) < 2000:
                        continue
                    mask_list.append(tmp_mask)
                    object_index += 1   
                except:
                    pass


           	

            # 把前面一步的mask_list里的元素填到预定义的mask数组里，从第一个通道开始填（第0个通道空出来）
            for i,tmp_mask in enumerate(mask_list):
                mask[:,:,i+1] = tmp_mask
            # 得到一张shape是(h, w)的mask，其中元素取值是0, 1, 2...
            mask = np.argmax(mask,axis = 2).astype(np.uint8)

            if len(mask_list) != 0:
                # 如果当前图片上存在满足要求的实例，则直接做数据增强就可以结束了
                frames_,masks_ = self.Augmentation(frame,mask)
            else:
                # 如果当前图片上一个满足要求的实例都没有，则需要做复制粘贴，
                # 把来自其他图片的满足要求的实例贴到当前图片上
                tmp_sample_num = random.randint(1,3)  # 尝试粘贴的实例数量
                sampled_objects = []
                max_iter = 20
                while tmp_sample_num > 0 and max_iter > 0:
                    max_iter -= 1
                    # 从数据集中随机采样一个实例，
                    # TODO: 这里也可以事先就把满足（边界框大小）条件的实例id给保存起来，这样就不用试错了
                    tmp = random.sample(self.anno_list,1)
                    if tmp[0]['bbox'][2] * tmp[0]['bbox'][3] < 3000:
                        continue
                    sampled_objects.append(tmp[0])
                    tmp_sample_num -= 1
            

            

                sampled_f_m = []
                for sampled_object in sampled_objects:
                    ob_img_path = os.path.join(self.image_dir,str(sampled_object['image_id']).zfill(12) + '.jpg')
                    ob_frame = np.array(Image.open(ob_img_path).convert('RGB'))
                    h,w,_ = ob_frame.shape
                    ob_segs = sampled_object['segmentation']
                    ob_bbox = sampled_object['bbox']  # (x1, y1, w, h)
                    ob_segs_list = []
                    try:
                        for ob_seg in ob_segs:
                            tmp = np.array(ob_seg).reshape(-1,2).astype(np.int32)
                            ob_segs_list.append(tmp)
                        ob_mask = np.zeros((h,w)).astype(np.uint8)
                        ob_mask = cv2.fillPoly(ob_mask, ob_segs_list,object_index)
                        object_index += 1

                        y1,y2 = int(ob_bbox[1]),int(ob_bbox[1] + ob_bbox[3])
                        x1,x2 = int(ob_bbox[0]),int(ob_bbox[0] + ob_bbox[2])
                        # 得到采样到的实例所对应的frame区域和mask区域
                        ob_mask = ob_mask[y1:y2,x1:x2]
                        ob_frame = ob_frame[y1:y2,x1:x2,:]
                        # 区域大小是由实例大小决定的, 裁剪出的区域大小等于2倍的save_h_w
                        save_h_w = int(math.sqrt(ob_bbox[3] ** 2 + ob_bbox[2] ** 2))
                        # np.lib.pad(array, pad_width, mode='constant', **kwargs)
                        ob_mask = np.lib.pad(ob_mask,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),
                                                      (int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2))),
                                             'constant',constant_values=0)
                        ob_frame = np.lib.pad(ob_frame,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),
                                                        (int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2)),
                                                        (0,0)),'constant',constant_values=0)

                        # cv2.imwrite('test.jpg',ob_frame)
                        # cv2.imwrite('test.png',ob_mask*255)
                        sampled_f_m.append([ob_frame,ob_mask])
                    except:
                        pass        
                frames_,masks_ = self.Augmentation(frame,mask,sampled_f_m)

        else:

            object_index = 1
            mask_list = []
            for inst in instances_list:
                segs = inst['segmentation']
                segs_list = []
                try:
                    for seg in segs:
                        if len(np.array(seg).shape) == 0:
                            continue
                        tmp = np.array(seg).reshape(-1,2).astype(np.int32)
                        segs_list.append(tmp)
                    tmp_mask = np.zeros((h,w))
                    tmp_mask = cv2.fillPoly(tmp_mask, segs_list,1)
                    if np.sum(tmp_mask) < 2000:
                        continue
                    mask_list.append(tmp_mask)
                    object_index += 1   
                except:
                    pass
            # 如果当前图像上满足要求的实例数量大于5了，则随机取5个
            if len(mask_list) > 5:
                mask_list = random.sample(mask_list,5)

            object_index = len(mask_list) + 1

            for i,tmp_mask in enumerate(mask_list):
                mask[:,:,i+1] = tmp_mask
            # 得到当前图像原生的mask， shape是(h, w), 取值是1, 2, 3,...
            mask = np.argmax(mask,axis = 2).astype(np.uint8)
            # 不管当前帧目前有多少个（不会超过5个）满足要求的实例，这里都再额外加1~3个来自其他帧的实例
            tmp_sample_num = random.randint(1,3)
            sampled_objects = []
            max_iter = 20
            while tmp_sample_num > 0 and max_iter > 0:
                max_iter -= 1
                tmp = random.sample(self.anno_list,1)
                if tmp[0]['bbox'][2] * tmp[0]['bbox'][3] < 3000:
                    continue
                sampled_objects.append(tmp[0])
                tmp_sample_num -= 1
            

            

            sampled_f_m = []
            for sampled_object in sampled_objects:
                ob_img_path = os.path.join(self.image_dir,str(sampled_object['image_id']).zfill(12) + '.jpg')
                ob_frame = np.array(Image.open(ob_img_path).convert('RGB'))
                h,w,_ = ob_frame.shape
                ob_segs = sampled_object['segmentation']
                ob_bbox = sampled_object['bbox']
                ob_segs_list = []
                try:
                    for ob_seg in ob_segs:
                        tmp = np.array(ob_seg).reshape(-1,2).astype(np.int32)
                        ob_segs_list.append(tmp)
                    ob_mask = np.zeros((h,w)).astype(np.uint8)
                    ob_mask = cv2.fillPoly(ob_mask, ob_segs_list,object_index)
                    object_index += 1

                    y1,y2 = int(ob_bbox[1]),int(ob_bbox[1] + ob_bbox[3])
                    x1,x2 = int(ob_bbox[0]),int(ob_bbox[0] + ob_bbox[2])
                    ob_mask = ob_mask[y1:y2,x1:x2]
                    ob_frame = ob_frame[y1:y2,x1:x2,:]

                    save_h_w = int(math.sqrt(ob_bbox[3] ** 2 + ob_bbox[2] ** 2))

                    ob_mask = np.lib.pad(ob_mask,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),
                                                  (int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2))),
                                         'constant',constant_values=0)
                    ob_frame = np.lib.pad(ob_frame,((int((save_h_w - ob_bbox[3])/2),int((save_h_w - ob_bbox[3])/2)),
                                                    (int((save_h_w - ob_bbox[2])/2),int((save_h_w - ob_bbox[2])/2)),
                                                    (0,0)),'constant',constant_values=0)

                    # cv2.imwrite('test.jpg',ob_frame)
                    # cv2.imwrite('test.png',ob_mask*255)
                    sampled_f_m.append([ob_frame,ob_mask])
                except:
                    pass        
            frames_,masks_ = self.Augmentation(frame,mask,sampled_f_m)

        num_object = 0
        ob_list = []
        # N_frames = np.empty((3,)+(384,384,)+(3,), dtype=np.float32)
        # N_masks = np.empty((3,)+(384,384,), dtype=np.uint8)
        for f in range(3):
            tmp_mask,num_object,ob_list = self.mask_process(masks_[f],f,num_object,ob_list)
            N_frames[f],N_masks[f] = frames_[f],tmp_mask

        Fs = torch.from_numpy(np.transpose(N_frames.copy(), (3, 0, 1, 2)).copy()).float()
        Ms = torch.from_numpy(self.All_to_onehot(N_masks).copy()).float()
        if num_object == 0:
            num_object += 1
        num_objects = torch.LongTensor([num_object])
        return Fs, Ms, num_objects, info


if __name__ == '__main__':
    import os
    import sys
    pwd = os.getcwd()
    sys.path.append(pwd)
    from utils.helpers import overlay_davis
    import matplotlib.pyplot as plt
    import pdb
    import argparse
    def get_arguments():
        parser = argparse.ArgumentParser(description="xxx")
        parser.add_argument("-o", type=str, help="", default='./tmp')
        parser.add_argument("-Dcoco", type=str, help="path to coco",default='/smart/haochen/cvpr/data/COCO/coco/')
        parser.add_argument("-Ddavis", type=str, help="path to davis",default='/smart/haochen/cvpr/data/DAVIS/')
        return parser.parse_args()
    args = get_arguments()
    davis_root = args.Ddavis
    coco_root = args.Dcoco
    output_dir = args.o
    dataset = Coco_MO_Train('{}train2017'.format(coco_root),'{}annotations/instances_train2017.json'.format(coco_root))
    palette = Image.open('{}Annotations/480p/blackswan/00000.png'.format(davis_root)).getpalette()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i,(Fs,Ms,num_objects,info) in enumerate(dataset):
        mask = np.argmax(Ms.numpy(), axis=0).astype(np.uint8)
        img_list = []
        for f in range(3):
            pF = (Fs[:,f].permute(1,2,0).numpy()*255.).astype(np.uint8)
            pE = mask[f]
            canvas = overlay_davis(pF, pE, palette)
            img = np.concatenate([pF,canvas],axis = 0)
            img_list.append(img)
        out_img = np.concatenate(img_list,axis = 1)
        out_img = Image.fromarray(out_img)
        print('saving images to {}'.format(output_dir))
        out_img.save(os.path.join(output_dir, str(i).zfill(5) + '.jpg'))
        pdb.set_trace()
