
import numpy as np
import cv2
import matplotlib.pyplot as plt


def plot_all_boxes(img, boxes):
    """Plots all rectangles from boxes onto img."""
    if type(boxes) != list:
        boxes = [boxes]
    copy = img.copy()
    alpha = 0.4
    for box in boxes:
       x, y, w, h = box.x, box.y, box.width, box.height
       rand_color = list(np.random.random(size=3) * 256)
       cv2.rectangle(copy, (x, y), (x+w, y+h), rand_color, -1)
    
    img_new = cv2.addWeighted(copy, alpha, img, 1-alpha, 0)
    return img_new
    

class BBox():
    """BBox object representing boundingrectangle. (x coord of top-left, y coord of top-left, wdith, height)"""
    # TODO: improve names of variables
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.y_bottom = y + height
        self.x_right = x + width



def segment(img, x=0, y=0):

    # Input: cv2 image of page. Output: BBox objects for content blocks in page

    MIN_TEXT_SIZE = 10
    HORIZONTAL_POOLING = 25
    img_width = img.shape[1]
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to create a binary image
    img_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)
    
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(img_bw, (7,7), 0) 
    
    # Morphological Gradient
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    m1 = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, k1)
    
    # Morphological Closing
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (HORIZONTAL_POOLING, 5))
    m2 = cv2.morphologyEx(m1, cv2.MORPH_CLOSE, k2)

    # Dilation
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    m3 = cv2.dilate(m2, k3, iterations=2)
    
    # Find contours in the processed image
    contours = cv2.findContours(m3, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    
    bboxes = []
    for c in contours:
        
        bx,by,bw,bh = cv2.boundingRect(c)
        
        # filter out bounding boxes that are too small
        if bh < MIN_TEXT_SIZE:
            continue

        bboxes.append(BBox(x, y+by, img_width, bh))        
    
    # sort bounding boxes by their y-coordinate
    return sorted(bboxes, key=lambda x: x.y)



def segment_page(img):
    """Segment page into content blocks."""
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding to create a binary image
    img_bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5)

    column_split_x = img.shape[1]//2 +2 # FIXME: hardcoded

    # Compute horizontal histogram
    lookup_distance = 20
    hist = img_bw[:, column_split_x-lookup_distance:column_split_x+lookup_distance]
    hist = hist.mean(axis=1)

    assert len(hist) == img.shape[0]

    # Make it binary
    # If is 255, then it's white, so set it to 0 (two columns)
    # If is not 255, then it's black, so set it to 1 (single column)
    hist = np.where(hist == 255, 0, 1)

    # DEBUG
    #plt.plot(hist)
    #plt.show()

    hist_diff = np.diff(hist)

    
    start_single_columns = np.argwhere(hist_diff == 1)
    end_single_columns = np.argwhere(hist_diff == -1)


    # An array of start and end y-locations of single columns sections
    single_columns = np.concatenate((start_single_columns, end_single_columns), axis=1)
    # Sort by start
    single_columns = single_columns[single_columns[:, 0].argsort()]

    # Merge them if the distance between the intervals defined by them is less than 5 or they overlap
    new_single_columns = []
    already_processed = []
    #print("Original single columns: ", single_columns)

    c = True
    while c:
        c = False
        for i in range(len(single_columns)):
            if i in already_processed:
                continue

            # If it's the last element, then add it to the new list
            if i == len(single_columns) - 1:
                new_single_columns.append((single_columns[i][0], single_columns[i][1]))
                already_processed.append(i)
                continue

            current_start = single_columns[i][0]
            current_end = single_columns[i][1]
            next_start = single_columns[i+1][0]
            next_end = single_columns[i+1][1]

            # Merge them if the distance between the intervals defined by them is less than 5 or they overlap
            if next_start - current_end < 100 or next_start <= current_end: # FIXME: hardcoded
                new_single_columns += [(min(current_start, next_start), max(current_end, next_end))]
                already_processed += [i, i+1]
                c = True

            # If they don't overlap, then add them to the new list
            else:
                new_single_columns.append((current_start, current_end))
                already_processed.append(i)
            

        single_columns = new_single_columns
        new_single_columns = []
        already_processed = []

    single_columns = np.array(single_columns)
    #print("New:", single_columns)

    #######################################################################################

    bboxes = []

    # Two columns for sure
    if (hist == 0).all():
        print("Two columns")
        bboxes += segment(img[:, :column_split_x], 0, 0)
        bboxes += segment(img[:, column_split_x:], column_split_x, 0)
        return bboxes

    # Single column for sure
    #if (hist == 1).all():
    #    print("Single column")
    #    bboxes += segment(img, 0, 0)
    #    return bboxes

    # If there are too many 1s or -1s, then it's probably a single column
    #if (hist_diff == 1).sum() / len(hist_diff) > 0.01: # FIXME: hardcoded
    #    print("Single column identified the hardway?", (hist_diff == 1).sum() / len(hist_diff))
    #    bboxes += segment(img, 0, 0)
    #    return bboxes

    #print("Idk", single_columns.tolist())

    # Otherwise, we have to figure out which is which
    else:
        current_y = 0
        for start, end in single_columns:
            # Convert from current y to the next y as two columns #
            if start != current_y:
                # Split the image into sections
                left_column = img[current_y:start, :column_split_x]
                right_column = img[current_y:start, column_split_x:]

                # Segment
                bboxes += segment(left_column, 0, current_y)
                bboxes += segment(right_column, column_split_x, current_y)

            # Convert from start to end as one column
            one_column = img[start:end, :]
            bboxes += segment(one_column, 0, start)

            # Update current y
            current_y = end

        if current_y < img.shape[0]:
            # Convert from current y to the end as two columns
            # split the image into sections
            left_column = img[current_y:, :column_split_x]
            right_column = img[current_y:, column_split_x:]

            # segment
            bboxes += segment(left_column, 0, current_y)
            bboxes += segment(right_column, column_split_x, current_y)
        
    # Merge bboxes
    new = []
    [new.append(box) for box in bboxes if not new or not any([box.y > box2.y and box.y_bottom < box2.y_bottom for box2 in new])]
    bboxes = new

    # Remove completely empty bboxes
    # For each bbox, check if it's compleatly white
    # If it is, then remove it
    new = []
    percentage_threshold = 0.9 # FIXME: hardcoded
    for box in bboxes:
        if (img[box.y:box.y_bottom, box.x:box.x_right] == 255).sum() / (box.width * box.height) > percentage_threshold:
            new.append(box)

    return bboxes


