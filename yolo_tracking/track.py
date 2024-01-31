# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import argparse
from functools import partial
from pathlib import Path
import copy
import datetime
import time

import torch
import numpy as np
import cv2
import os
import pandas as pd

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils.checks import TestRequirements
from examples.detectors import get_yolo_inferer

__tr = TestRequirements()
__tr.check_packages(('ultralytics @ git+https://github.com/mikel-brostrom/ultralytics.git', ))  # install

from ultralytics import YOLO
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.plotting import save_one_box

from examples.utils import write_mot_results

from drive_utils.wrapper import DriveHandler

##############################
# For Geofencing + Counter
##############################
class Counter:
    def __init__(self, x1, y1, x2, y2, idx):
        """
        Initialize a counter

        Args:
            roi = (x1, x2, y1, y2) which have been normalized to [0,1] range
            x1, y1 ---------------
            |                    |
            |         ROI        |
            |                    |
            --------------- x2, y2

            idx = which camera
        """
        
        self.roi_x1 = x1
        self.roi_y1 = y1
        self.roi_x2 = x2
        self.roi_y2 = y2
        self.idx = idx
        
        self.steps = 0

        self.drive_handler = DriveHandler()

        self.reset()

    def reset(self):
        '''
        Definition of IN and OUT
        
        IN -> the person has entered the room
        OUT -> the person is in the geofencing zone, before entering the room
        
        buffer_in -> to store person who first appeared in the room
        buffer_out -> to store person who first appeared in the geofencing zone, before entering
        '''
        self.move_in = {}
        self.move_out = {} # not implemented yet
        self.count_in = 0
        self.count_out = 0 # not implemented yet
        
        self.buffer_out = {} # to store id of ppl that haven enter yet     
        self.buffer_in = {}  # to store id of ppl that first appear inside

        self.current_date = datetime.datetime.now().date()
        self.current_hour = datetime.datetime.now().hour

        # create new logfile at local
        self.logfile = f'log/camera{str(self.idx).zfill(3)}_{self.current_date.strftime("%Y-%m-%d")}_count.txt'
        if not os.path.isdir('log'):
            os.mkdir('log')
        if not os.path.isfile(self.logfile):
            with open(self.logfile, 'w') as f:
                # this will create an empty logile
                pass
        else:
            # if the logfile existed, means our code has been interrupted and restarted
            # fill in the missing values between this duration
            # read back the last count
            self.resume()

        # update ytd data to google drive
        try:
            self.drive_handler.post()
            pass
        except:
            print('Google API daily quota reached. Will upload tomorrow after quota renewed')

    def resume(self):
        df = pd.read_csv(self.logfile,delimiter = ' ',header = None, engine = 'python')
        df.columns = ['Date','Time','Count']
        df_hr = pd.to_datetime(df['Time'].to_list(),format='%H:%M:%S.%f').hour  
        num_rows , _ = df.shape
    
        # print(f"the number of rows is {num_rows}")
        # print(df_hr[-1])
        # print(self.current_hour)
    
        if num_rows < self.current_hour-1:
            print("filling missing values")
            with open (self.logfile,'a') as f:
                missHr = self.current_hour-df_hr[-1]
                lastCount = int(df.iloc[-1]['Count'])
                                        
                for i in range(missHr):
                    missTime = datetime.time(df_hr[num_rows-1]+i+1,0,0,1)
                    datetimeInfo = datetime.datetime.combine(datetime.datetime.today().date(),missTime)
                    f.write(f'{datetimeInfo} {lastCount}\n')
                    print("written")
                    
        # update self.count_in = lastCount (before system interrupted and restart)
        self.count_in = lastCount            
                                
    def clear_buffer(self):
        # increment steps
        self.steps += 1

        # every 900 steps (assuming 30fps, then is every 30s)
        now = datetime.datetime.now()
        if self.steps % 300:
            # remove IDs in buffer_in 
            remove_ids = []
            for id, timestamp in self.buffer_in.items():
                delta_time = now - timestamp
                
                # remove the id in buffer_id, if the last appearance is more than 30s
                if delta_time.total_seconds() > 30:
                    remove_ids.append(id)
            
            for id in remove_ids:
                del self.buffer_in[id]
                
            # remove IDs in buffer_out
            remove_ids = []
            for id, timestamp in self.buffer_out.items():
                delta_time = now - timestamp
                
                # remove the id in buffer_id, if the last appearance is more than 30s
                if delta_time.total_seconds() > 30:
                    remove_ids.append(id)
            
            for id in remove_ids:
                del self.buffer_out[id]                
        
    def update(self, img_shape=None, pred_boxes=None):
        """
        Update the total number of objects move in/out the ROI

        Args:
            img_shape: the img shape
            pred_boxes: the bbox of predicted obj
        """

        # Update Detect results
        if pred_boxes:
            for d in reversed(pred_boxes):
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                xyxy = d.xyxy.squeeze().cpu().detach().numpy()
                x1, y1, x2, y2 = xyxy

                # centroid
                x_mid = (x1 + x2) / 2
                y_mid = (y1 + y2) / 2
                
                # conditions
                condition_1 = x_mid >= self.roi_x1 * img_shape[1]
                condition_2 = x_mid <= self.roi_x2 * img_shape[1]
                condition_3 = y_mid >= self.roi_y1 * img_shape[0]
                condition_4 = y_mid <= self.roi_y2 * img_shape[0]
                within_roi = condition_1 and condition_2 and condition_3 and condition_4
                
                # update count
                if id is None:
                    return None # tracker will assign None as ID, if the track is not alive long enough
                    
                if (within_roi) and (id not in self.buffer_in.keys()):
                    self.buffer_out[id] = datetime.datetime.now()
                elif (within_roi) and (id in self.buffer_in.keys()):
                    pass
                elif (not within_roi) and (id not in self.buffer_out.keys()):
                    self.buffer_in[id] = datetime.datetime.now()
                elif (not within_roi) and (id in self.buffer_out.keys()):
                    self.count_in += 1
                    self.move_in[id] = datetime.datetime.now()
                    del self.buffer_out[id]
        
        # clear buffer
        # self.clear_buffer()

    def log(self):
        # Get the current date and time
        now = datetime.datetime.now()

        # Check if the current time is at the start of a new hour
        if now.hour != self.current_hour:
            # update current hour
            print(now.strftime("%Y-%m-%d %H:%M:%S"))
            self.current_hour = now.hour

            # log total count
            with open(self.logfile, 'a') as f:
                f.write(f'{datetime.datetime.now()} {self.count_in}\n')

        # Check if a new day has passed
        if now.date() > self.current_date:
            # reset
            self.current_date = now.date()
            self.reset()
            
# overwrite ultralytics.engine.predictor.BasePredictor
def write_results(self, idx, results, batch):
    """Write inference results to a file or directory."""
    p, im, _ = batch
    log_string = ''
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    if self.source_type.webcam or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
        log_string += f'{idx}: '
        frame = self.dataset.count
    else:
        frame = getattr(self.dataset, 'frame', 0)
    self.data_path = p
    self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
    log_string += '%gx%g ' % im.shape[2:]  # print string
    result = results[idx]
    log_string += result.verbose()

    result_boxes = copy.deepcopy(result.boxes)
    if self.args.save or self.args.show:  # Add bbox to image
        plot_args = {
            'line_width': self.args.line_width,
            'boxes': self.args.boxes,
            'conf': self.args.show_conf,
            'labels': self.args.show_labels}
        if not self.args.retina_masks:
            plot_args['im_gpu'] = im[idx]
        self.plotted_img = result.plot(**plot_args)
        
    # update move in count
    if self.counters is not None:
        # update counters
        self.counters[idx].update(self.plotted_img.shape, result_boxes)
        
        # get the roi bbox points
        img_shape = self.plotted_img.shape
        x1 = int(self.counters[idx].roi_x1 * img_shape[1])
        y1 = int(self.counters[idx].roi_y1 * img_shape[0])
        x2 = int(self.counters[idx].roi_x2 * img_shape[1])
        y2 = int(self.counters[idx].roi_y2 * img_shape[0])
        
        # draw roi
        pts = [[x1,y1],[x1,y2],[x2,y2],[x2,y1]]
        pts = np.array(pts, int)
        pts = pts.reshape((-1, 1, 2))
        self.plotted_img = cv2.polylines(self.plotted_img, [pts], True, (0,0,255), 5)
        
        # put text
        text = f'in: {self.counters[idx].count_in}'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = int(self.plotted_img.shape[0] * 0.002)
        font_thickness = 2
        origin = (int(self.plotted_img.shape[0]*0.35), int(self.plotted_img.shape[1]*0.5))
        x, y = origin
        text_color = (255, 255, 255)
        text_color_bg = (0, 0, 0)
        
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        
        cv2.rectangle(self.plotted_img, origin, (x + text_w, y + text_h), text_color_bg, -1)
        cv2.putText(self.plotted_img,text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # log
        self.counters[idx].log()
    
    # Write
    if self.args.save_txt:
        result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
    if self.args.save_crop:
        result.save_crop(save_dir=self.save_dir / 'crops',
                         file_name=self.data_path.stem + ('' if self.dataset.mode == 'image' else f'_{frame}'))

    return log_string                        
##############################
# END SECTION
##############################

                   
def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert predictor.custom_args.tracking_method in TRACKERS, \
        f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = \
        ROOT /\
        'boxmot' /\
        'configs' /\
        (predictor.custom_args.tracking_method + '.yaml')
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class
        )
        # motion only modeles do not have
        if hasattr(tracker, 'model'):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):

    yolo = YOLO(
        args.yolo_model if 'yolov8' in str(args.yolo_model) else 'yolov8n.pt',
    )

    results = yolo.track(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        show=args.show,
        stream=True,
        device=args.device,
        show_conf=args.show_conf,
        save_txt=args.save_txt,
        show_labels=args.show_labels,
        save=args.save,
        verbose=args.verbose,
        exist_ok=args.exist_ok,
        project=args.project,
        name=args.name,
        classes=args.classes,
        imgsz=args.imgsz,
        vid_stride=args.vid_stride,
        line_width=args.line_width
    )

    yolo.add_callback('on_predict_start', partial(on_predict_start, persist=True))    
    
    if 'yolov8' not in str(args.yolo_model):
        # replace yolov8 model
        m = get_yolo_inferer(args.yolo_model)
        model = m(
            model=args.yolo_model,
            device=yolo.predictor.device,
            args=yolo.predictor.args
        )
        yolo.predictor.model = model

    # store custom args in predictor
    yolo.predictor.custom_args = args


    ##############################
    # GEOFENCING + Counter
    ##############################
    yolo.predictor.counters = None
    if args.roi_xyxys is not None:
        yolo.predictor.counters = []
        roi_xyxys = args.roi_xyxys.split('][')
        for i in range(len(roi_xyxys)):
            xyxy = roi_xyxys[i].replace('[', '').replace(']', '')
            xyxy = xyxy.split(',')
            xyxy = [float(item) for item in xyxy]
            x1, y1, x2, y2  = xyxy
            yolo.predictor.counters.append(Counter(x1, y1, x2, y2, i+1))
            
    import types
    yolo.predictor.write_results = types.MethodType(write_results, yolo.predictor)
    ##############################
    # END SECTION
    ##############################
    
    for frame_idx, r in enumerate(results):

        if r.boxes.data.shape[1] == 7:

            if yolo.predictor.source_type.webcam or args.source.endswith(VID_FORMATS):
                p = yolo.predictor.save_dir / 'mot' / (args.source + '.txt')
                yolo.predictor.mot_txt_path = p
            elif 'MOT16' or 'MOT17' or 'MOT20' in args.source:
                p = yolo.predictor.save_dir / 'mot' / (Path(args.source).parent.name + '.txt')
                yolo.predictor.mot_txt_path = p

            if args.save_mot:
                write_mot_results(
                    yolo.predictor.mot_txt_path,
                    r,
                    frame_idx,
                )

            if args.save_id_crops:
                for d in r.boxes:
                    print('args.save_id_crops', d.data)
                    save_one_box(
                        d.xyxy,
                        r.orig_img.copy(),
                        file=(
                            yolo.predictor.save_dir / 'crops' /
                            str(int(d.cls.cpu().numpy().item())) /
                            str(int(d.id.cpu().numpy().item())) / f'{frame_idx}.jpg'
                        ),
                        BGR=True
                    )

    if args.save_mot:
        print(f'MOT results saved to {yolo.predictor.mot_txt_path}')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', type=Path, default=WEIGHTS / 'yolov8n',
                        help='yolo model path')
    parser.add_argument('--reid-model', type=Path, default=WEIGHTS / 'osnet_x0_25_msmt17.pt',
                        help='reid model path')
    parser.add_argument('--tracking-method', type=str, default='deepocsort',
                        help='deepocsort, botsort, strongsort, ocsort, bytetrack')
    parser.add_argument('--source', type=str, default='0',
                        help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640],
                        help='inference size h,w')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.7,
                        help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--project', default=ROOT / 'runs' / 'track',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='video frame-rate stride')
    parser.add_argument('--show-labels', action='store_false',
                        help='either show all or only bboxes')
    parser.add_argument('--show-conf', action='store_false',
                        help='hide confidences when show')
    parser.add_argument('--save-txt', action='store_true',
                        help='save tracking results in a txt file')
    parser.add_argument('--save-id-crops', action='store_true',
                        help='save each crop to its respective id folder')
    parser.add_argument('--save-mot', action='store_true',
                        help='save tracking results in a single txt file')
    parser.add_argument('--line-width', default=None, type=int,
                        help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--per-class', default=False, action='store_true',
                        help='not mix up classes when tracking')
    parser.add_argument('--verbose', default=True, action='store_true',
                        help='print results per frame')
    parser.add_argument('--vid_stride', default=1, type=int,
                        help='video frame-rate stride')
    parser.add_argument('--roi-xyxys', type=str, default=None,
                        help='x1y1x2y2 of RoI (in range 0 to 1), i.e.: [0.3,0.5,0.3,0.5] OR [0.3,0.5,0.3,0.5][0, 1, 0.5, 0.5]')
                        
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    run(opt)
