#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 14:34:56 2017

@author: m0r00ds
"""

class Model(object):

  def load_data(self):
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_placeholders(self):
    raise NotImplementedError("Each Model must re-implement this method.")

  def create_feed_dict(self, input_batch, label_batch):
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_model(self, input_data):
    raise NotImplementedError("Each Model must re-implement this method.")

  def add_loss_op(self, pred):
    raise NotImplementedError("Each Model must re-implement this method.")

  def run_epoch(self, sess, input_data, input_labels):
    raise NotImplementedError("Each Model must re-implement this method.")

  def fit(self, sess, input_data, input_labels):
    raise NotImplementedError("Each Model must re-implement this method.")

  def predict(self, sess, input_data, input_labels=None):
    raise NotImplementedError("Each Model must re-implement this method.")