# !/usr/bin/env python
# -- coding: utf-8 --
# @Time : 2020/6/16 10:57
# @Author : liumin
# @File : tensorboard.py
from torch.utils.tensorboard import SummaryWriter

class DummyWriter(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                     flush_secs=120, filename_suffix=''):
        super().__init__(log_dir, comment, purge_step, max_queue,flush_secs, filename_suffix)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        super(DummyWriter, self).add_scalar(tag, scalar_value, global_step, walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        super(DummyWriter, self).add_scalars(main_tag, tag_scalar_dict, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        super(DummyWriter, self).add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        super(DummyWriter, self).add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_graph(self, model, input_to_model=None, verbose=False):
        super(DummyWriter, self).add_graph(model, input_to_model, verbose)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        super(DummyWriter, self).add_images(tag, img_tensor, global_step, walltime, dataformats)

    def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,
                             walltime=None, rescale=1, dataformats='CHW', labels=None):
        super(DummyWriter, self).add_image_with_boxes(tag, img_tensor, box_tensor, global_step,
                             walltime)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        super(DummyWriter, self).add_figure(tag, figure, global_step, close, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        super(DummyWriter, self).add_text(tag, text_string, global_step, walltime)

    def add_onnx_graph(self, onnx_model_file):
        super(DummyWriter, self).add_onnx_graph(onnx_model_file)

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None):
        super(DummyWriter, self).add_embedding(mat, metadata, label_img, global_step, tag, metadata_header)

    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):
        super(DummyWriter, self).add_pr_curve(tag, labels, predictions, global_step,num_thresholds, weights, walltime)

    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None):
        super(DummyWriter, self).add_mesh(tag, vertices, colors, faces, config_dict, global_step, walltime)

    def close(self):
        super(DummyWriter, self).close()

    def flush(self):
        super(DummyWriter, self).flush()

