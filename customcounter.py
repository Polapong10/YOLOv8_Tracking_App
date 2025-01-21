import cv2
from collections import defaultdict

from ultralytics.utils.plotting import Annotator, colors

class ObjectCounter:
    def __init__(self):

        # Image and Annotation information
        self.im0 = None
        self.tf = None
        
        # Model information
        self.names = None

        # Region Information
        self.reg_pts = None
        self.region_color = (255, 0, 255)
        
        # Object counting
        self.ext_road = 0
        self.ent_road = 0
        self.counting_list = []

        # Tracks info
        self.track_history = defaultdict(list)
        self.draw_tracks = False

    def set_arguments(self,
                      classes_names,
                      reg_pts,
                      line_thickness = 3,
                      draw_tracks=False
                      ):
        self.names = classes_names
        self.tf = line_thickness
        self.reg_pts = reg_pts
        self.draw_tracks = draw_tracks
    
    # Custom tracking
    def extract_tracking(self, tracks):
        # Create anonotator 
        self.annotator = Annotator(self.im0, self.tf, self.names)
        for i in range(0, len(self.reg_pts) - 1, 2):
            self.im0 = cv2.line(self.im0, self.reg_pts[i], self.reg_pts[i + 1], self.region_color, 5)

        # if tracks[0].boxes.id is None it means that there are no objects detected
        if tracks[0].boxes.id is None:
            pass
        else:
            boxes = tracks[0].boxes.xyxy.cpu()
            classes = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            for box, track_id, cls in zip(boxes, track_ids, classes):
                self.annotator.box_label(box, label = f'#{track_id}: {self.names[cls]}', color = colors(int(cls), True))  # Draw bounding box

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
                # Keep only the last 30 points
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw tracks if draw_tracks is True
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(track_line,
                                                            color = (255, 255, 255),
                                                            track_thickness = 1)
                
                # if track_line[0][-1] < self.reg_pts[0][0] and track_line[-1][1] >= self.reg_pts[0][1] + 2:
                # 0 is x and 1 is y for entrance road
                if track_line[-1][0] > self.reg_pts[0][0] and track_line[-1][1] <= self.reg_pts[1][1] + 2:
                    if track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        self.ent_road += 1
                if track_line[-1][0] <= self.reg_pts[3][0] and track_line[-1][1] >= self.reg_pts[3][1]:
                    if track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        self.ext_road += 1

        entroad_label = 'Entrance Road : ' + f'{self.ent_road}'
        extroadroad_label = 'Exist Road : ' + f'{self.ext_road}'
        # Draw count label
        self.annotator_countlabel(label = entroad_label, p1 = (20, 5))
        self.annotator_countlabel(label = extroadroad_label, p1 = (20, 65))
    
    # Function for draw count label
    def annotator_countlabel(self, label, p1 = (5, 5), color = (255, 255, 255), txt_color = (0, 0, 0)):
        tl = self.tf or round(0.002 * (self.im.shape[0] + self.im.shape[1]) / 2) + 1
        tf = max(tl - 1, 1)
        gap = int(24 * tl)

        # Get text size for in_count and out_count
        w, h = cv2.getTextSize(str(label), 0, fontScale=tl / 3, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        self.im0 = cv2.rectangle(self.im0, p1, p2, color, -1, cv2.LINE_AA)  # filled
        self.im0 = cv2.putText(self.im0,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    tl / 3,
                    txt_color,
                    thickness = 1,
                    lineType=cv2.LINE_AA)
        
    # Function for start counting
    def start_counting(self, im0, tracks):
        """
        Main function to start the object counting process.

        Args:
            im0 (ndarray): Current frame from the video stream.
            tracks (list): List of tracks obtained from the object tracking process.
        """
        self.im0 = im0  # store image
        self.extract_tracking(tracks)
        return self.im0