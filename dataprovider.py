from __future__ import division

import numpy as np
import cPickle as pickle
import os, sys
import scipy.io

class dataprovider(object):
    def __init__(self, train_list, test_list, img_feat_dir, global_feat_dir, sen_dir, vocab_size, knowledge='hard',
        val_list='', phrase_len=5, batch_size=20, seed=1):
        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list
        self.img_feat_dir = img_feat_dir
        self.global_feat_dir = global_feat_dir
        self.sen_dir = sen_dir
        self.phrase_len = phrase_len
        self.cur_id = 0
        self.epoch_id = 0
        self.num_prop = 100
        self.img_feat_size = 4096
        self.num_test = 1000
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.is_save = False
        self.knowledge = knowledge
        np.random.seed(seed)
        self.train_id_list = np.random.permutation(len(train_list))

    def _reset(self):
        self.cur_id = 0
        self.train_id_list = np.random.permutation(len(self.train_list))
        self.is_save = False

    def _read_single_feat(self, img_id):
        # img_id = self.train_list[self.train_id_list[self.cur_id]]

        sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
        pos_ids = np.array(sen_feat['pos_id']).astype('int')
        pos_ind = np.where(pos_ids != -1)[0]

        if len(pos_ind) > 0:
            img_feat = np.zeros((self.num_prop, self.img_feat_size))
            cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id))

            cur_feat_norm = np.sqrt((cur_feat*cur_feat).sum(axis=1))
            cur_feat /= cur_feat_norm.reshape(cur_feat.shape[0], 1)

            img_feat_global = np.load('%s/%d.npy'%(self.global_feat_dir, img_id))
            img_feat_global = np.tile(img_feat_global, (self.num_prop, 1)).astype('float')

            img_feat[:cur_feat.shape[0], :self.img_feat_size] = cur_feat
            img_feat = img_feat.astype('float')

            sens = sen_feat['sens']
            sen_id = np.random.randint(len(pos_ind))
            # print img_id, sen_id
            sen = sens[pos_ind[sen_id]]
            if len(sen) > self.phrase_len:
                sen = sen[:self.phrase_len]

            # pad sen tokens to phrase_len with UNK token as (self.vocab_size-1)
            sen_token = np.ones(self.phrase_len, dtype=int)*(self.vocab_size-1)     
            dec_token = np.ones(self.phrase_len, dtype=int)*(self.vocab_size-1)    
            indicator = np.zeros(self.phrase_len, dtype=int)
            sen_token[:len(sen)] = sen
            dec_token[:-1] = sen_token[1:]

            indicator[:len(sen)] = 1
            if self.knowledge == 'hard':
                kbp = sen_feat['q_dist'][pos_ind[sen_id]]
            elif self.knowledge == 'coco':
                kbp = sen_feat['q_dist_soft_coco'][pos_ind[sen_id]]
            else:
                kbp = sen_feat['q_dist_soft_pas'][pos_ind[sen_id]]
            bbx_vlv = sen_feat['bbx_vlv']
            y = pos_ids[pos_ind[sen_id]]
            return img_feat, img_feat_global, sen_token, dec_token, indicator, kbp, bbx_vlv, y
        else:
            return None, None, None, None, None, None, None, -1

    def get_next_batch(self):
        img_feat_batch = np.zeros((self.batch_size, self.num_prop, self.img_feat_size), dtype=float)
        img_feat_global_batch = np.zeros((self.batch_size, self.num_prop, self.img_feat_size), dtype=float)
        token_batch = np.zeros((self.batch_size, self.phrase_len), dtype=int)
        dec_batch = np.zeros((self.batch_size, self.phrase_len), dtype=int)
        mask_batch = np.zeros((self.batch_size, self.phrase_len), dtype=int)
        kbpv_batch = np.zeros((self.batch_size, self.num_prop), dtype=float)
        kbpl_batch = np.ones((self.batch_size, self.num_prop), dtype=float)
        bbx_vlv_batch = np.zeros((self.batch_size, self.num_prop, 4), dtype=float)
        y_batch = np.zeros(self.batch_size).astype('int')
        num_cnt = 0

        while num_cnt < self.batch_size:
            if self.cur_id == len(self.train_list):
                self._reset()
                self.epoch_id += 1
                self.is_save = True
                print('Epoch %d complete'%(self.epoch_id))
            img_id = self.train_list[self.train_id_list[self.cur_id]]        
            img_feat, img_feat_global, sen_token, dec_token, indicator, kbp, bbx_vlv, y = self._read_single_feat(img_id)
            if y != -1:
                img_feat_batch[num_cnt] = img_feat
                img_feat_global_batch[num_cnt] = img_feat_global
                token_batch[num_cnt] = sen_token
                y_batch[num_cnt] = y
                dec_batch[num_cnt] = dec_token
                mask_batch[num_cnt] = indicator
                
                if self.knowledge == 'hard':
                    kbpv_batch[num_cnt] = kbp
                    if not np.all(kbp == 0):
                        kbpl_batch[num_cnt] = kbp
                else:
                    kbpv_batch[num_cnt] = kbp / (np.max(kbp)+1e-6)
                    if not np.all(kbp < 0.3):
                        kbpl_batch[num_cnt] = kbp / np.max(kbp)
                bbx_vlv_batch[num_cnt][:len(bbx_vlv)] = bbx_vlv
                num_cnt += 1
            self.cur_id += 1   

        return img_feat_batch, img_feat_global_batch, token_batch, dec_batch, mask_batch, kbpv_batch, kbpl_batch, bbx_vlv_batch, y_batch

    def get_test_feat(self, img_id):
        sen_feat = np.load('%s/%d.pkl'%(self.sen_dir, img_id))
        pos_ids = np.array(sen_feat['pos_id']).astype('int')
        pos_ind = np.where(pos_ids != -1)[0]
        gt_pos_all = sen_feat['gt_pos_all']
        gt_bbx_all = sen_feat['gt_box']     # ground truth bbx for query: [xmin, ymin, xmax, ymax]
        if self.knowledge == 'hard':
            kbpv = sen_feat['q_dist']
        elif self.knowledge == 'coco':
            kbpv = sen_feat['q_dist_soft_coco']
        else:
            kbpv = sen_feat['q_dist_soft_pas']
        num_sample = len(pos_ids)
        num_corr = 0
        gt_h = sen_feat['height'][0]
        gt_w = sen_feat['width'][0]
        bbx_pos = sen_feat['ss_box'].astype('float')

        if len(pos_ids) > 0:
            img_feat = np.zeros((self.num_prop, self.img_feat_size)).astype('float')
            cur_feat = np.load('%s/%d.npy'%(self.img_feat_dir, img_id)).astype('float')

            cur_feat_norm = np.sqrt((cur_feat*cur_feat).sum(axis=1))
            cur_feat /= cur_feat_norm.reshape(cur_feat.shape[0], 1)
            
            img_feat_global = np.load('%s/%d.npy'%(self.global_feat_dir, img_id))
            img_feat_global = np.tile(img_feat_global, (self.num_prop, 1))

            img_feat[:cur_feat.shape[0], :self.img_feat_size] = cur_feat

            sen_feat_batch = np.zeros((len(pos_ids), self.phrase_len)).astype('int')
            mask_batch = np.zeros((len(pos_ids), self.phrase_len)).astype('int')
            kbpl_batch = np.ones((len(pos_ids), self.num_prop), dtype=float)

            gt_batch = []
            gt_pos = []

            sens = sen_feat['sens']
            for sen_ind in range(len(pos_ids)):
                cur_sen = sens[sen_ind]
                sen_token = np.ones(self.phrase_len)*(self.vocab_size-1)
                sen_token = sen_token.astype('int')
                if len(cur_sen) > self.phrase_len:
                    cur_sen = cur_sen[:self.phrase_len]
                sen_token[:len(cur_sen)] = cur_sen
                sen_feat_batch[sen_ind] = sen_token
                mask_batch[sen_ind][:len(cur_sen)] = 1
                if self.knowledge == 'hard':
                    if not np.all(kbpv[sen_ind] == 0):
                        kbpl_batch[sen_ind] = kbpv[sen_ind].astype('float')
                else:
                    if not np.all(kbpv[sen_ind] < 0.3):
                        kbpl_batch[sen_ind] = kbpv[sen_ind]/np.max(kbpv[sen_ind])
                gt_batch.append(gt_pos_all[sen_ind])

                if not np.any(gt_bbx_all[sen_ind]):
                    num_sample -= 1

            gt_pos = np.array(gt_bbx_all, dtype=float)
            return img_feat, img_feat_global, sen_feat_batch, mask_batch, kbpl_batch, gt_batch, bbx_pos, gt_pos, gt_h, gt_w, num_sample
        else:
            return None, None, None, None, None, None, None, None, 0, 0, 0

if __name__ == '__main__':
    train_list = []
    test_list = []
    img_feat_dir = '~/dataset/flickr30k_img_bbx_cyc_vgg_det'
    sen_dir = '~/dataset/flickr30k_img_sen_feat_cyc'
    vocab_size = 17150
    with open('../flickr30k_test.lst') as fin:
        for img_id in fin.readlines():
            test_list.append(int(img_id.strip()))
    train_list = np.array(train_list).astype('int')
    cur_dataset = dataprovider(train_list, test_list, img_feat_dir, sen_dir, vocab_size)
    for i in range(10000):
        # img_feat_batch, token_batch, dec_batch, mask_batch, kbpv_batch, kbpl_batch, bbx_vlv_batch, y_batch = cur_dataset.get_next_batch()
        img_feat_batch, sen_feat_batch, mask_batch, kbpl_batch, \
        gt_batch, bbx_pos, gt_pos, gt_h, gt_w, num_sample = cur_dataset.get_test_feat(test_list[cur_dataset.cur_id])
        cur_dataset.cur_id = (cur_dataset.cur_id+1)%len(test_list)
        print img_feat_batch.shape#, token_batch.shape, enc_batch.shape, dec_batch.shape, mask_batch.shape  
        print '%d/%d'%(cur_dataset.cur_id, len(cur_dataset.test_list))
