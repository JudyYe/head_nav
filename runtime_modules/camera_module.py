import numpy as np
import cv2
import redis
import pickle


class OpenCVCamera:
    def __init__(self, outsize=[352, 352]):
        self.cap = cv2.VideoCapture(0)
        self.outsize = outsize
        self.goal = None

    def get_frame(self):
        ret, img = self.cap.read()
        if not ret:
            raise Exception("Camera not connected or channel not available")
        h, w = img.shape[:2]
        pad = abs(h - w) // 2
        if h > w:
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # resize image
        img = cv2.resize(img, self.outsize, interpolation=cv2.INTER_LINEAR)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print("Selected:",2*float(x-self.outsize[0]//2)/self.outsize[0], 2*float(y-self.outsize[1]//2)/self.outsize[1])
            self.goal = [2*float(x-self.outsize[0]//2)/self.outsize[0], 2*float(y-self.outsize[1]//2)/self.outsize[1]]

    def goal_selector(self, frame_skip=30):
        for _ in range(frame_skip):
            img = self.get_frame()
        while self.goal is None:
            cv2.imshow("Select Goal", img)
            cv2.setMouseCallback('Select Goal', self.click)
            cv2.waitKey(1)
        cv2.destroyAllWindows()

class RedisCamera:
    # def __init__(self, outsize=[352, 352], h_lim_in_480=[322, 479], w_lim_in_640=[192, 502], debug=False):
    def __init__(self, outsize=[352, 352], h_lim_in_480=[300, 479], w_lim_in_640=[50, 600], debug=False):

        self.redis_client = redis.Redis(host='192.168.123.222', port=6379, db=0)
        self.last_id = '$'
        self.outsize = outsize
        self.goal = None
        self.h_lim_in_480 = h_lim_in_480
        self.w_lim_in_640 = w_lim_in_640
        self.debug = debug

        self.in_max_size = 640
        self.pad_offset = [0, 0]

        self.manip_zoomed_img_size = None

        self.debug = False
        # self.debug = True


        self.h_lower_bound = 0
        self.h_upper_bound = 300
        self.w_lower_bound = 420
        self.w_upper_bound = 1080
    
    def find_normalized_goal_in_zoomed_frame(self, goal_yx_normalized):
        # row is x, column is y --- follow the convention of numpy, not opencv
        yx_in_640_square = (goal_yx_normalized * 0.5 + 0.5) * self.in_max_size
        xy_in_640_square = np.array([yx_in_640_square[1], yx_in_640_square[0]])  # (640, 640)
        xy_in_raw_img = xy_in_640_square - np.array(self.pad_offset)    # (640, 640)
        x_in_zoomed_img = int(xy_in_raw_img[0]) - self.h_lim_in_480[0]
        y_in_zoomed_img = int(xy_in_raw_img[1]) - self.w_lim_in_640[0]
        h_of_zoomed_img = self.h_lim_in_480[1] - self.h_lim_in_480[0]
        w_of_zoomed_img = self.w_lim_in_640[1] - self.w_lim_in_640[0]

        # compute the scaled coordinates
        x_in_zoomed_img = float(x_in_zoomed_img / h_of_zoomed_img) * 2 - 1
        y_in_zoomed_img = float(y_in_zoomed_img / w_of_zoomed_img) * 2 - 1
        # compose the goal in the zoomed image
        
        self.goal = [y_in_zoomed_img, x_in_zoomed_img]
        return np.array(self.goal)



    def check_if_goal_in_overlap_region(self, goal_yx_in_pixel_space):
        # goal_2D_in_pixel_space lives in self.outsize -- [640, 640]
        goal_yx_in_640 = np.array(goal_yx_in_pixel_space) / self.outsize[0] * self.in_max_size
        goal_xy_in_640 = np.array([goal_yx_in_640[1], goal_yx_in_640[0]])
        # print("goal_xy_in_640", goal_xy_in_640)

        # import pdb; pdb.set_trace()
        goal_2D_in_raw = goal_xy_in_640 - np.array(self.pad_offset)
        # print("goal_2D_in_raw", goal_2D_in_raw)
        # check if the goal is in the overlap region
        x_in_range = self.h_lim_in_480[0] <= goal_2D_in_raw[0] <= self.h_lim_in_480[1]
        y_in_range = self.w_lim_in_640[0] <= goal_2D_in_raw[1] <= self.w_lim_in_640[1]
        # import pdb; pdb.set_trace()
        if x_in_range and y_in_range:
            return True
        else:
            return False
        # if goal_2D_in_raw[0] < self.w_lim_in_640[0] or goal_2D_in_raw[0] > self.w_lim_in_640[1]:
            # return False
    

    def refresh_goal_yx_in_raw_manip_cam_frame(self, goal_yx_normalized):
        x_in_raw_manip_img = self.h_lower_bound + goal_yx_normalized[1] / self.outsize[1] * self.manip_zoomed_img_size[0]
        y_in_raw_manip_img = self.w_lower_bound + goal_yx_normalized[0] / self.outsize[0] * self.manip_zoomed_img_size[1]

        x_in_raw_manip_img = int(x_in_raw_manip_img)
        y_in_raw_manip_img = int(y_in_raw_manip_img)
        
        goal_yx_in_raw_manip_cam_frame = np.array([y_in_raw_manip_img, x_in_raw_manip_img])
        self.redis_client.set("goal_opencv_xy_in_raw_manip_cam_frame", pickle.dumps(goal_yx_in_raw_manip_cam_frame))
        return goal_yx_in_raw_manip_cam_frame
    
    def get_manip_cam_frame(self, scale_factor=0.4):
        raw_image_data = self.redis_client.get("manip_camera_stream")    
        depth = pickle.loads(
                                self.redis_client.get("manip_camera_depth")
                            )        
        # Retrieve the JPEG bytes from the message
        jpeg_bytes = raw_image_data
        # Convert bytes to a numpy array and decode the image
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #print("img shape", img.shape)
        # apply scale factor
        img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)), interpolation=cv2.INTER_LINEAR)

        self.h_lower_bound = 0
        self.h_upper_bound = (img.shape[0] - 1)
        self.w_lower_bound = 0
        self.w_upper_bound = (img.shape[1] - 1)
        return img, depth



    def get_zoomed_manip_cam_frame_and_depth(self, i_for_transition=0, i_min_for_transition=0, i_max_for_transition=479):
        raw_image_data = self.redis_client.get("manip_camera_stream")            
        # Retrieve the JPEG bytes from the message
        jpeg_bytes = raw_image_data
        # Convert bytes to a numpy array and decode the image
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        manip_raw_img = img.copy()

        raw_img_shape = img.shape
        h_zoomed_lower_bound = 0
        h_zoomed_upper_bound = 300
        h_final_bound = [0, raw_img_shape[0] - 1]

        w_final_bound = [0, raw_img_shape[1] - 1]
        w_zoomed_lower_bound = 420
        w_zoomed_upper_bound = 1080

        if i_for_transition < i_min_for_transition:

            transition_process = 0
        elif i_for_transition < i_max_for_transition:
            transition_process = (i_for_transition - i_min_for_transition) / (i_max_for_transition - i_min_for_transition)
        else:
            transition_process = 1


        h_lower_bound = int(h_zoomed_lower_bound + (h_final_bound[0] - h_zoomed_lower_bound) * transition_process)
        h_upper_bound = int(h_zoomed_upper_bound + (h_final_bound[1] - h_zoomed_upper_bound) * transition_process)
        w_lower_bound = int(w_zoomed_lower_bound + (w_final_bound[0] - w_zoomed_lower_bound) * transition_process)
        w_upper_bound = int(w_zoomed_upper_bound + (w_final_bound[1] - w_zoomed_upper_bound) * transition_process)
        self.h_lower_bound = h_lower_bound
        self.h_upper_bound = h_upper_bound
        self.w_lower_bound = w_lower_bound
        self.w_upper_bound = w_upper_bound
        zoomed_img = img[ h_lower_bound:h_upper_bound, w_lower_bound:w_upper_bound]  # 1280, 72


        self.manip_zoomed_img_size = zoomed_img.shape
        zoomed_img = cv2.resize(zoomed_img, self.outsize, interpolation=cv2.INTER_LINEAR)

        # reshape the img to ()
        return zoomed_img, manip_raw_img

    def get_navi_raw_img_with_padding(self):
        raw_image_data = self.redis_client.get("camera_stream")            
        # Retrieve the JPEG bytes from the message
        jpeg_bytes = raw_image_data
        # Convert bytes to a numpy array and decode the image
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        h, w = img.shape[:2]    # 480, 640
        pad = abs(h - w) // 2   # pad = 80

        if h > w:
            self.pad_offset = [0, pad]
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            self.pad_offset = [pad, 0]
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        return img


    def get_zoomed_frame(self, x, y, i_for_transition=0, i_min_for_transition=0, i_max_for_transition=100):
        img = self.get_navi_raw_img_with_padding()

        if i_for_transition < i_min_for_transition:
            transition_process = 0.3
        elif i_for_transition < i_max_for_transition:
            transition_process = 0.3 + 0.7* (i_for_transition - i_min_for_transition) / (i_max_for_transition - i_min_for_transition)
        else:
            transition_process = 1

        # Center with the input x, y and zoom in the image by ratio of i_for_transition
        # When transition_process = 0, the zoomed image should center at the input x, y
        # When transition_process = 1, the image should be the whole image

        h, w = img.shape[:2]    # 480, 640
        pad = abs(h - w) // 2   # pad = 80

        if h > w:
            self.pad_offset = [0, pad]
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            self.pad_offset = [pad, 0]
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        zoomed_img_h = 2 * int( max(y, h - y) * transition_process)
        zoomed_img_w = 2 * int( max(x, w - x) * transition_process)

        print("zoomed_img_h", zoomed_img_h)
        print("zoomed_img_w", zoomed_img_w)
        print(" i_for_transition", i_for_transition)
        print("transition_process", transition_process)

        h_lower_bound = max(0, int(y - zoomed_img_h / 2))
        h_upper_bound = min(h, int(y + zoomed_img_h / 2))
        w_lower_bound = max(0, int(x - zoomed_img_w / 2))
        w_upper_bound = min(w, int(x + zoomed_img_w / 2))

        # based on y, h_lower_bound and h_upper_bound to scale y to [-1 1]
        scaled_y = float(y - h_lower_bound) / (h_upper_bound - h_lower_bound) * 2 - 1
        # based on x, w_lower_bound and w_upper_bound to scale x to [-1 1]
        scaled_x = float(x - w_lower_bound) / (w_upper_bound - w_lower_bound) * 2 - 1
        
        self.goal_on_zoomed_img = [scaled_x, scaled_y]


        # import pdb; pdb.set_trace()
        zoomed_img = img[ h_lower_bound:h_upper_bound, w_lower_bound:w_upper_bound]  # (160, 430, 3)
        
        
        # make the zoomed in image square
        zoomed_img = cv2.resize(zoomed_img, self.outsize, interpolation=cv2.INTER_LINEAR)
        return zoomed_img

    def get_frame(self):
        raw_image_data = self.redis_client.get("camera_stream")            
        # Retrieve the JPEG bytes from the message
        jpeg_bytes = raw_image_data
        
        # Convert bytes to a numpy array and decode the image
        np_arr = np.frombuffer(jpeg_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        h, w = img.shape[:2]    # 480, 640
        pad = abs(h - w) // 2   # pad = 80

        self.in_max_size = max(h, w)

        #assert self.debug == True, "Debug mode is not enabled. Please enable it to visualize the goal selection region."
        
        if self.debug:
            alpha = 0.2
            overlay = img.copy()
            radius = 2
            half_radius = radius // 2
            # refer to https://stackoverflow.com/questions/25642532/opencv-pointx-y-represent-column-row-or-row-column
            for x in range(self.h_lim_in_480[0]-half_radius, self.h_lim_in_480[1]+half_radius):
                for y in range(self.w_lim_in_640[0]-half_radius, self.w_lim_in_640[1]+half_radius):
                    cv2.circle(overlay, (y, x), radius, (0, 255, 0), -1)
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        if h > w:
            self.pad_offset = [0, pad]
            img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            self.pad_offset = [pad, 0]
            img = cv2.copyMakeBorder(img, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # resize image
        img = cv2.resize(img, self.outsize, interpolation=cv2.INTER_LINEAR)
        return img
    
    def click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print("Selected:",2*float(x-self.outsize[0]//2)/self.outsize[0], 2*float(y-self.outsize[1]//2)/self.outsize[1])
            # self.goal = [2*float(x-self.outsize[0]//2)/self.outsize[0], 2*float(y-self.outsize[1]//2)/self.outsize[1]]
            self.raw_goal = [x, y]
            self.goal = [2*float(x-self.raw_img_with_padding_size[0]//2)/self.raw_img_with_padding_size[0], 2*float(y-self.raw_img_with_padding_size[1]//2)/self.raw_img_with_padding_size[1]]
            
    def goal_selector(self, frame_skip=30):
        for _ in range(frame_skip):
            img = self.get_navi_raw_img_with_padding()
        self.raw_img_with_padding_size = img.shape[:2]
        while self.goal is None:
            cv2.imshow("Select Goal", img)
           
            cv2.setMouseCallback('Select Goal', self.click)
            

            cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = OpenCVCamera()
    camera.goal_selector()
    while True:
        img = camera.get_frame()
        cv2.imshow("Camera", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break