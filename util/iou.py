import numpy as np
import os, sys

def calc_iou(box1, box2):
	# box: [xmin, ymin, xmax, ymax]
	iou = 0.0
	if box1[2] <= box1[0] or box1[3] <= box1[1]:
		return iou
	if box2[2] <= box2[0] or box2[3] <= box2[1]:
		return iou		
	if box1[2] <= box2[0] or box1[0] >= box2[2]:
		return iou
	if box1[3] <= box2[1] or box1[1] >= box2[3]:
		return iou

	xl_min = min(box1[0], box2[0])
	xl_max = max(box1[0], box2[0])
	xr_min = min(box1[2], box2[2])
	xr_max = max(box1[2], box2[2])

	yl_min = min(box1[1], box2[1])
	yl_max = max(box1[1], box2[1])
	yr_min = min(box1[3], box2[3])
	yr_max = max(box1[3], box2[3])

	inter = float(xr_min-xl_max)*float(yr_min-yl_max)
	union = float(xr_max-xl_min)*float(yr_max-yl_min)

	iou = float(inter) / float(union)
	if iou < 0:
		iou = 0.0
	return iou

def calc_iou_by_reg_feat(gt, pred, reg_feat, ht, wt):
	# gt, pred: [xmin, ymin, xmax, ymax]
	# reg_feat:	[t_x, t_y, t_w, t_h]
	# t_x = (x_gt - x_pred) / w_pred (center point x)
	# t_y = (y_gt - y_pred) / h_pred (center point y)
	# t_w = log(w_gt / w_pred)
	# t_h = log(h_gt / h_pred)

	pred = np.array(pred).astype('float')
	pred_w = pred[2]-pred[0]+1.0
	pred_h = pred[3]-pred[1]+1.0
	reg_w = np.exp(reg_feat[2])*pred_w-1.0
	reg_h = np.exp(reg_feat[3])*pred_h-1.0

	if reg_w > 0.0 and reg_h > 0.0:
		reg = np.zeros(4).astype('float32')
		reg[0] = (pred[0]+pred[2])/2.0+pred_w*reg_feat[0]-reg_w/2.0
		reg[1] = (pred[1]+pred[3])/2.0+pred_h*reg_feat[1]-reg_h/2.0
		reg[2] = (pred[0]+pred[2])/2.0+pred_w*reg_feat[0]+reg_w/2.0
		reg[3] = (pred[1]+pred[3])/2.0+pred_h*reg_feat[1]+reg_h/2.0
		reg[0] = max(0.0, reg[0])
		reg[1] = max(0.0, reg[1])
		reg[2] = min(wt, reg[2])
		reg[3] = min(ht, reg[3])
		return calc_iou(gt, reg), reg
	else:
		return 0.0, None

def calc_iou_cyc(gt, reg_feat, ht, wt):
	# gt, pred: [xmin, ymin, xmax, ymax]
	# reg_feat t: 4d vector
	# t[0] = 2*xmin/w - 1
	# t[1] = 2*ymin/h - 1
	# t[2] = 2*xmax/w - 1
	# t[3] = 2*ymax/h - 1

	reg = np.zeros(4).astype('float32')
	if reg_feat[0] < reg_feat[2] and reg_feat[1] < reg_feat[3]:
		reg[0] = (reg_feat[0]+1.0)*wt/2.0
		reg[1] = (reg_feat[1]+1.0)*ht/2.0
		reg[2] = (reg_feat[2]+1.0)*wt/2.0
		reg[3] = (reg_feat[3]+1.0)*ht/2.0
		reg[0] = max(0.0, reg[0])
		reg[1] = max(0.0, reg[1])
		reg[2] = min(wt, reg[2])
		reg[3] = min(ht, reg[3])

		reg_iou = calc_iou(gt, reg)
		return reg_iou, reg
	else:
		return 0.0, reg
		