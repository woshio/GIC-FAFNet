import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('/media/yg/winOs/GIC/ultralytics-main/GIC-FAFNet.yaml')
    model.train(data='/home/yg/ultralytics-main-6-19/Dior.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                #close_mosaic=0,
                workers=8, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0',
                optimizer='SGD', # using SGD
                #patience=0, # set 0 to close earlystop.
                #resume=True, # 断点续训,YOLO初始化时选择last.pt,例如YOLO('last.pt')
                #amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                #amp=False
                )