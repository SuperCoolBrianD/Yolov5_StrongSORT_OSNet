#!/usr/bin/env python
__copyright__ = """
Copyright (c) 2020 Tananaev Denis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions: The above copyright notice and this permission
notice shall be included in all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import sys
sys.path.append('/lidar_dynamic_objects_detection/detection_3d')
import argparse
import os
import tensorflow as tf
from voxel_parameters import Parameters
from voxel_detection_dataset import DetectionDataset
from voxel_preprocess_util import get_voxels_grid
from voxel_yolo_model import YoloV3_Lidar
from voxel_training_helpers import (
    setup_gpu,
    initialize_model,
    load_model,
    get_optimizer,
)
from voxel_losses import detection_loss
from voxel_summary_helpers import train_summaries, epoch_metrics_summaries
from voxel_metrics import EpochMetrics
from tqdm import tqdm



def train_step(param_settings, train_samples, model, optimizer, epoch_metrics=None):

    with tf.GradientTape() as tape:
        top_view, box_grid, _ = train_samples
        predictions = model(top_view, training=True)
        (
            obj_loss,
            label_loss,
            z_loss,
            delta_xy_loss,
            width_loss,
            height_loss,
            delta_orient_loss,
        ) = detection_loss(box_grid, predictions, num_classes=5)
        losses = [
            obj_loss,
            label_loss,
            z_loss,
            delta_xy_loss,
            width_loss,
            height_loss,
            delta_orient_loss,
        ]
        total_detection_loss = tf.reduce_sum(losses)

        # Get L2 losses for weight decay
        total_loss = total_detection_loss + tf.math.add_n(model.losses)
    gradients = tape.gradient(total_loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if epoch_metrics is not None:
        epoch_metrics.train_loss(total_detection_loss)

    train_outputs = {
        "total_loss": total_loss,
        "losses": losses,
        "box_grid": box_grid,
        "predictions": predictions,
        "top_view": top_view,
    }

    return train_outputs


def val_step(samples, model, epoch_metrics=None):

    top_view, box_grid, _ = samples
    predictions = model(top_view, training=False)
    (
        obj_loss,
        label_loss,
        z_loss,
        delta_xy_loss,
        width_loss,
        height_loss,
        delta_orient_loss,
    ) = detection_loss(box_grid, predictions, num_classes=5)
    losses = [
        obj_loss,
        label_loss,
        z_loss,
        delta_xy_loss,
        width_loss,
        height_loss,
        delta_orient_loss,
    ]
    total_detection_loss = tf.reduce_sum(losses)
    # print(total_detection_loss)
    if epoch_metrics is not None:
        epoch_metrics.val_loss(total_detection_loss)


def train(resume=False):
    setup_gpu()
    # General parameters
    param = Parameters()

    # Init label colors and label names
    tf.random.set_seed(param.settings["seed"])

    train_dataset = DetectionDataset(
        param.settings,
        "train.datatxt",
        augmentation=param.settings["augmentation"],
        shuffle=True,
    )

    param.settings["train_size"] = train_dataset.num_samples
    val_dataset = DetectionDataset(param.settings, "val.datatxt", shuffle=False)
    param.settings["val_size"] = val_dataset.num_samples

    model = YoloV3_Lidar(weight_decay=param.settings["weight_decay"])
    voxels_grid = get_voxels_grid(
        param.settings["voxel_size"], param.settings["grid_meters"]
    )
    input_shape = [param.settings["batch_size"], voxels_grid[0], voxels_grid[1], 2]
    initialize_model(model, input_shape)
    model.summary()
    start_epoch, model = load_model(param.settings["checkpoints_dir"], model, resume)
    model_path = os.path.join(param.settings["checkpoints_dir"], "{model}-{epoch:04d}")

    learning_rate, optimizer = get_optimizer(
        param.settings["optimizer"],
        param.settings["scheduler"],
        train_dataset.num_it_per_epoch,
    )
    epoch_metrics = EpochMetrics()

    for epoch in range(start_epoch, param.settings["max_epochs"]):
        save_dir = model_path.format(model=model.name, epoch=epoch)
        epoch_metrics.reset()
        for train_samples in tqdm(
            train_dataset.dataset,
            desc=f"Epoch {epoch}",
            total=train_dataset.num_it_per_epoch,
        ):
            train_outputs = train_step(
                param.settings, train_samples, model, optimizer, epoch_metrics
            )
            train_summaries(train_outputs, optimizer, param.settings, learning_rate)
        print(train_outputs['total_loss'])
        for val_samples in tqdm(
            val_dataset.dataset, desc="Validation", total=val_dataset.num_it_per_epoch
        ):
            val_step(val_samples, model, epoch_metrics)

        epoch_metrics_summaries(param.settings, epoch_metrics, epoch)
        epoch_metrics.print_metrics()
        # Save all
        param.save_to_json(save_dir)
        epoch_metrics.save_to_json(save_dir)
        model.save(save_dir)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Train CNN.")
    # parser.add_argument(
    #     "--resume",
    #     type=lambda x: x,
    #     nargs="?",
    #     const=True,
    #     default=False,
    #     help="Activate nice mode.",
    # )
    # args = parser.parse_args()
    train(resume=True)
