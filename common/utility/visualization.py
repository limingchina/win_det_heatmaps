import os
import cv2
import logging
import numpy as np
import matplotlib.pyplot as plt
from common.utility.utils import float2int

FONT_STYLE = cv2.FONT_HERSHEY_PLAIN

def plot_LearningCurve(train_loss, valid_loss, log_path, jobName):
    '''
    Use matplotlib to plot learning curve at the end of training
    train_loss & valid_loss must be 'list' type
    '''
    plt.figure(figsize=(12, 5))
    plt.title("Learning Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    epochs = np.arange(len(train_loss))
    plt.plot(epochs, np.array(train_loss), 'r', label='train')
    plt.plot(epochs, np.array(valid_loss), 'b', label='valid')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(log_path, jobName + '.png'))


def debug_vis(img, window_corner, label=None, raw_img=None, plotLine=True):
    if isinstance(img, str):
        if os.path.exists(img):
            cv_img_patch_show = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        cv_img_patch_show = img.copy()
    else:
        assert 0, "unKnown Type of img in debug_vis"

    flag5 = False
    if len(window_corner) == 4:
        left_top, left_bottom, right_bottom, right_top = window_corner
    elif len(window_corner) == 5:
        left_top, left_bottom, right_bottom, right_top, center = window_corner
        flag5 = True
    else:
        assert 0

    num_windows = len(left_top)
    for idx in range(num_windows):
        cv2.putText(cv_img_patch_show,'1',float2int(left_top[idx]), FONT_STYLE, 1, (255,0,0), 1)
        cv2.putText(cv_img_patch_show,'2',float2int(left_bottom[idx]), FONT_STYLE, 1, (0,255,0), 1)
        cv2.putText(cv_img_patch_show,'3',float2int(right_bottom[idx]), FONT_STYLE, 1, (0,0,255), 1)
        cv2.putText(cv_img_patch_show,'4',float2int(right_top[idx]), FONT_STYLE, 1, (0,255,255), 1)

        cv2.circle(cv_img_patch_show, float2int(left_top[idx]), 3, (255, 0, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(left_bottom[idx]), 3, (0, 255, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(right_bottom[idx]), 3, (0, 0, 255), -1)
        cv2.circle(cv_img_patch_show, float2int(right_top[idx]), 3, (0, 255, 255), -1)

        if flag5:
            cv2.putText(cv_img_patch_show, '5', float2int(center[idx]), FONT_STYLE, 1, (255, 255, 0), 1)
            cv2.circle(cv_img_patch_show, float2int(center[idx]), 3, (255, 255, 0), -1)

        if plotLine:
            thickness = 2
            color = (50, 250, 50)
            cv2.line(cv_img_patch_show, float2int(left_top[idx]), float2int(left_bottom[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(left_bottom[idx]), float2int(right_bottom[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(right_bottom[idx]), float2int(right_top[idx]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(right_top[idx]), float2int(left_top[idx]), color, thickness)

    # ----------- vis label --------------
    if isinstance(label, np.ndarray):
        label_ = label.copy() * 255.0
        empty = np.ones((10, cv_img_patch_show.shape[1], 3), dtype=cv_img_patch_show.dtype)*255
        label_to_draw = np.hstack((label_[0], label_[1], label_[2], label_[3])).astype(cv_img_patch_show.dtype)
        label_to_draw = cv2.cvtColor(label_to_draw, cv2.COLOR_GRAY2BGR)
        cv_img_patch_show = np.vstack((cv_img_patch_show, empty, label_to_draw))

    cv2.imshow('patch', cv_img_patch_show)
    cv2.waitKey(0)

def is_ccw(points):
    # Compute the signed area
    area = 0
    for i in range(len(points) - 1):
        area += points[i][0] * points[i + 1][1] - points[i + 1][0] * points[i][1]
    area += points[-1][0] * points[0][1] - points[0][0] * points[-1][1]
    # Normally it should be area > 0. However, since the Y direction of an image is
    # downward, the check should be flipped.
    return area < 0

def vis_eval_result(img, window, plotLine=False, saveFilename=None):
    if isinstance(img, str):
        if os.path.exists(img):
            cv_img_patch_show = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        cv_img_patch_show = img.copy()
    else:
        assert 0, "unKnown Type of img in debug_vis"
    logger = logging.getLogger()

    logger.info(f'image name: {img}')
    # pylint: disable=C0200
    for idx in range(len(window)):
        lt, lb, rb, rt = window[idx]['position'][:4]

        # Check if lt, lb, rb, rt forms a rectangle in counter-clockwise order
        if not is_ccw([lt, lb, rb, rt]):
            logger.warning(f'Index {idx} is not in counter-clockwise order: {window[idx]}')
            continue;

        # The result for resnet has some wrong window shapes. Let's correct them here.
        # The principle is to shift points to the minimum window shape.
        # We allow the maximum difference between the X components to be 0.3 times of
        # the width of the window. For the difference between the Y components, we allow
        # it to be 0.2 times of the height of the window since the difference between
        # Y components is likely to be smaller for windows.
        window_width = min(abs(lb[0] - rb[0]), abs(lt[0] - rt[0]))

        # If the X component of rt is smaller than that of lt, it's self-intersecting.
        # Move rt to align with rb.
        if rt[0] < lt[0] + 0.5:
            rt[0] = rb[0]
            logger.warning(f"Adjust rt[0] to rb[0]: index {idx}, {window[idx]}")

        # If the difference between the X component of the left points is larger than
        # 0.3 times of the width, adjust the point with smaller X component to
        # align with the point with the bigger X component.
        if abs(lt[0] - lb[0]) > 0.3 * window_width:
            if(lt[0] > lb[0]):
                lb[0] = lt[0]
                logger.warning(f"Adjust lb[0] to rb[0]: index {idx}, {window[idx]}")
            else:
                lt[0] = lb[0]
                logger.warning(f"Adjust rb[0] to lb[0]: index {idx}, {window[idx]}")

        # If the X component of rb is smaller than that of rt, it's self-intersecting.
        # Move rb to align with rt.
        if rb[0] < lb[0] + 0.5:
            rb[0] = rt[0]
            logger.warning(f"Adjust rb[0] to rt[0]: index {idx}, {window[idx]}")

        # If the difference between the X component of the right points is larger than
        # 0.3 times of the width, adjust the point with bigger X component to
        # align with the point with the smaller X component.
        if abs(rt[0] - rb[0]) > 0.3 * window_width:
            if(rt[0] < rb[0]):
                rb[0] = rt[0]
                logger.warning(f"Adjust rb[0] to rt[0]: index {idx}, {window[idx]}")
            else:
                rt[0] = rb[0]
                logger.warning(f"Adjust rt[0] to rb[0]: index {idx}, {window[idx]}")

        window_height = min(abs(lb[1] - lt[1]), abs(rb[1] - rt[1]))
        # If the difference between the Y component of the top points is larger than
        # 0.2 times of the height, adjust the point with smaller Y component to
        # align with the point with the bigger Y component.
        if abs(lt[1] - rt[1]) > 0.2 * window_height:
            if(lt[1] > rt[1]):
                logger.warning(f"Adjust rt[1] to lt[1]: index {idx}, {window[idx]}")
                rt[1] = lt[1]
            else:
                logger.warning(f"Adjust lt[1] to rt[1]: index {idx}, {window[idx]}")
                lt[1] = rt[1]

        # Similarly, if the difference between the Y component of bottom points is larger
        # than 0.2 times of the height, adjust the point with bigger Y component to
        # align with the point with smaller Y component.
        if abs(lb[1] - rb[1]) > 0.2 * window_height:
            if(lb[1] < rb[1]):
                logger.warning(f"Adjust rb[1] to lb[1]: index {idx}, {window[idx]}")
                rb[1] = lb[1]
            else:
                logger.warning(f"Adjust lb[1] to rb[1]: index {idx}, {window[idx]}")
                lb[1] = rb[1]

        # if the difference btween the two top points or the two bottom points is too large,
        # ignore the window
        if abs(lt[1] - rt[1]) > 2 * window_height or abs(lb[1] - rb[1]) > 2 * window_height:
            logger.warning(f"Ignore too large Y difference: index {idx}, {window[idx]}")
            continue

        # After all adjustments, check if lt, lb, rb, rt forms a rectangle in counter-clockwise
        # order again
        if not is_ccw([lt, lb, rb, rt]):
            logger.warning(f'Index {idx} is not in counter-clockwise order: {window[idx]}')
            continue;

        # Leave some debugging code here
        #if idx == 12 and "00155" in img:
        #print(idx, 'lt:', lt, 'lb:', lb, 'rb:', rb, 'rt:', rt)
        #print(f'window: {window[idx]}')

        # Shift the Y component of the left top point to the top by 5 pixels and put the idx
        # as a label for debugging purpose
        shifted_lt = lt
        shifted_lt[1] = shifted_lt[1] - 5
        cv2.putText(cv_img_patch_show, str(idx), float2int(shifted_lt[:2]), FONT_STYLE, 1.2, (255, 0, 0), 1)

        if plotLine:
            thickness = 3
            color = (50, 250, 50)
            cv2.line(cv_img_patch_show, float2int(lt[:2]), float2int(lb[:2]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(lb[:2]), float2int(rb[:2]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(rb[:2]), float2int(rt[:2]), color, thickness)
            cv2.line(cv_img_patch_show, float2int(rt[:2]), float2int(lt[:2]), color, thickness)

        cv2.circle(cv_img_patch_show, float2int(lt[:2]), 3, (255, 0, 0), -1)
        cv2.circle(cv_img_patch_show, float2int(lb[:2]), 3, (128, 200, 50), -1)
        cv2.circle(cv_img_patch_show, float2int(rb[:2]), 3, (0, 0, 255), -1)
        cv2.circle(cv_img_patch_show, float2int(rt[:2]), 3, (0, 255, 255), -1)

    if saveFilename != None:
        dirname = os.path.dirname(saveFilename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(os.path.join(saveFilename), cv_img_patch_show)
    else:
        cv2.imshow('Vis Evaluation Result', cv_img_patch_show)
        cv2.waitKey(0)


def vis_eval_result_with_gt(img, predWindow, gtWindow, plotLine=False, saveFilename=None):
    if isinstance(img, str):
        if os.path.exists(img):
            cv_img_patch_show = cv2.imread(img)
    elif isinstance(img, np.ndarray):
        cv_img_patch_show = img.copy()
    else:
        assert 0, "unKnown Type of img in debug_vis"

    predColor = (0,255,0)
    gtColor = (0,0,255)
    kptRadius = 3
    kptThickness = -1

    # GT
    for idx in range(len(gtWindow)):
        lt, lb, rb, rt = gtWindow[idx]

        if plotLine:
            lineThickness = 2
            color = (50, 50, 250)
            cv2.line(cv_img_patch_show, float2int(lt[:2]), float2int(lb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(lb[:2]), float2int(rb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rb[:2]), float2int(rt[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rt[:2]), float2int(lt[:2]), color, lineThickness)

        cv2.circle(cv_img_patch_show, float2int(lt[:2]), 3, gtColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(lb[:2]), 3, gtColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rb[:2]), 3, gtColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rt[:2]), 3, gtColor, kptThickness)

    # PRED
    for idx in range(len(predWindow)):
        lt, lb, rb, rt = predWindow[idx]['position'][:4]
        score = predWindow[idx]['score']

        if plotLine:
            lineThickness = 2
            color = (50, 250, 50)
            cv2.line(cv_img_patch_show, float2int(lt[:2]), float2int(lb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(lb[:2]), float2int(rb[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rb[:2]), float2int(rt[:2]), color, lineThickness)
            cv2.line(cv_img_patch_show, float2int(rt[:2]), float2int(lt[:2]), color, lineThickness)

        cv2.circle(cv_img_patch_show, float2int(lt[:2]), kptRadius, predColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(lb[:2]), kptRadius, predColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rb[:2]), kptRadius, predColor, kptThickness)
        cv2.circle(cv_img_patch_show, float2int(rt[:2]), kptRadius, predColor, kptThickness)

        cv2.putText(cv_img_patch_show, '%.2f' % score,
                    float2int(lt[:2]), FONT_STYLE, 1, (0, 255, 255), 1)

    if saveFilename != None:
        dirname = os.path.dirname(saveFilename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        cv2.imwrite(os.path.join(saveFilename), cv_img_patch_show)
    else:
        cv2.imshow('Vis Evaluation Result', cv_img_patch_show)
        cv2.waitKey(0)